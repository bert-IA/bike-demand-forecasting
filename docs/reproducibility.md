# Reproducibility and Traceability

## 🎯 Contexte : Pourquoi la traçabilité ?

Imagine : vous lancez un benchmark, obtenez des résultats → **mais 6 mois plus tard, comment refaire exactement les mêmes résultats ?**

### Problèmes sans traçabilité
- ❌ Quel commit du code a généré ces résultats ?
- ❌ Quelle version de polars/numpy était installée ?
- ❌ Le dataset a-t-il changé ?
- ❌ Quel config.yaml exact a été utilisé ?

### Première étape : capturer des métadonnées
La date, les informations Git, la configuration, les empreintes des fichiers
et les versions des distributions visibles aident à comprendre une exécution.
Elles ne suffisent pas à garantir sa reproductibilité : il faut notamment
conserver les fichiers correspondants et tenir compte du système utilisé.

---

## 📋 Module `tracing.py` : Vue d'ensemble

**Composition :**

```
ExecutionMetadata (classe)
    ↓ (créée par)
capture_metadata(..., project_root)
    ↓ (utilise)
  ├─→ get_git_info()
  ├─→ is_git_dirty()
  ├─→ get_package_versions()
  ├─→ get_installed_distributions()
  ├─→ hash_file_sha256()
  └─→ [lit les fichiers]
    ↓ (exportée par)
save_metadata()
```

---

## 🔧 Explications détaillées par fonction

### **1️⃣ `get_git_info(project_root)` → `(commit_hash, branch_name)`**

#### Enjeu
- Identifier le commit et la branche associés à l'exécution.
- Le commit ne contient pas les modifications locales non committées : il
  ne suffit donc pas à retrouver le code réellement exécuté.

#### Notions importantes
- **Git commit hash** = identifiant d'un instantané enregistré par Git.
  La fonction conserve actuellement ses huit premiers caractères.
- **Branch name** = branche actuelle (ex: `main`, `feat/tracing`)
- **subprocess.run()** = exécuter des commandes shell depuis Python

#### Approche
```python
# Appeler : git rev-parse HEAD → conserver les huit premiers caractères
# Appeler : git rev-parse --abbrev-ref HEAD → nom de branche
# Exécuter les deux commandes dans project_root.
```

#### À savoir sur `subprocess`
```python
result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=project_root,
    capture_output=True,
    text=True,
    check=True,
    timeout=5,
)
# result.stdout = output de la commande (string)
# check=True provoque une exception si la commande échoue
```

---

### `is_git_dirty(project_root)` → `bool`

Cette fonction indique si Git détecte des modifications locales, y compris les
fichiers non suivis, avec `git status --porcelain --untracked-files=normal`.
Elle retourne `True` si la sortie n'est pas vide. Une erreur Git est remontée
à l'appelant.

Les fichiers ignorés par Git ne sont pas pris en compte : `False` ne garantit
pas que les données ou d'autres fichiers ignorés sont identiques à ceux d'une
exécution précédente. Un état `True` signale des changements, mais ne
conserve pas leur contenu.

---

### **2️⃣ `get_package_versions()` → `dict[str, str]`**

#### Enjeu
- **Polars 0.19 vs 0.20 peuvent donner des résultats différents**
- Tracer exactement quelles versions ont été utilisées
- Reproductibilité scientifique

#### Notions importantes
- Une **distribution** est un paquet installé ; son nom peut différer du nom
  de module utilisé dans un `import`. Par exemple, la distribution `PyYAML`
  fournit le module `yaml`.
- `importlib.metadata.version()` lit la version de la distribution installée
  sans importer son module.
- Si la distribution est absente, la fonction enregistre `"not installed"`.

---

### `get_installed_distributions()` → `list[dict[str, str]]`

Cette fonction recense les noms et versions des distributions Python visibles
par l'interpréteur au moment de la capture. Elle trie le résultat par nom puis
par version. Une distribution visible n'est pas forcément utilisée par le
benchmark, et son inventaire ne décrit pas les bibliothèques système.

---

### `hash_file_sha256(path)` → `str`

Cette fonction calcule une empreinte SHA-256 à partir des octets d'un fichier,
en le lisant par morceaux. Elle retourne une chaîne hexadécimale de
64 caractères. Deux fichiers dont les empreintes correspondent ont très
probablement le même contenu ; une empreinte ne permet pas de retrouver le
fichier si celui-ci a été perdu.

Le benchmark enregistre les empreintes du fichier Parquet et de `uv.lock`.
L'empreinte est calculée au moment de la capture : les fichiers ne doivent pas
changer pendant l'exécution pour qu'elle décrive fidèlement les entrées
utilisées. Une erreur de lecture est remontée à l'appelant.

---

### **3️⃣ `capture_metadata()` → `ExecutionMetadata`**

#### Enjeu
- **L'orchestrateur** qui rassemble TOUT en une seule instance
- C'est la fonction "clé" : elle compose les autres

#### Notions importantes
- **Timestamp** = `datetime.now().isoformat()` : date et heure locales au
  format ISO, actuellement sans indication de fuseau horaire.
- **Config file content** = lire le fichier YAML brut (validation = déjà faite ailleurs)
- **Data shape analysis** = pas juste `n_rows`, mais aussi :
  - Nombre de **séries uniques** (`data["series_id"].unique().len()` en Polars)
  - Plage de dates (min/max de `timestamp`)
- La fonction enregistre aussi le chemin du fichier de données et le contenu
  texte de la configuration. Les colonnes ne sont pas actuellement listées.

---

### **4️⃣ `save_metadata()` → `None`**

#### Enjeu
- **Persister les métadonnées** dans un fichier JSON lisible
- Pouvoir les consulter des mois plus tard, dans n'importe quel éditeur

#### Notions importantes
- **dataclass → dict** : `asdict()` du module `dataclasses`
- **JSON indent** = `json.dumps(..., indent=2)` pour pretty-print
- **Path.write_text()** = écrire un string dans un fichier

#### Approche
```python
from dataclasses import asdict

data_dict = asdict(metadata)
json_str = json.dumps(data_dict, indent=2)
output_path.write_text(json_str)
# Résultat : fichier lisible, facile à versionner en audit
```

---

## 🧪 Exemple de métadonnées

Cet exemple est illustratif : les valeurs varient selon l'exécution.

```json
{
  "timestamp": "2026-11-25T09:35:52.003850",
  "git_commit": "4318afe4",
  "git_branch": "feat/benchmark-traceability",
  "git_dirty": true,
  "python_version": "3.12.3",
  "packages": {
    "polars": "2.0.0",
    "statsmodels": "0.15.0",
    "numpy": "2.5.4",
    "PyYAML": "6.0.3"
  },
  "config_file": "configs/benchmark_bike_validation.yaml",
  "config_content": "...",
  "lockfile_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "installed_distributions": [
    {"name": "numpy", "version": "2.5.4"},
    {"name": "PyYAML", "version": "6.0.3"}
  ],
  "data_sha256": "937e2615694f54c2b2b1694b6f144c30880f6879d52ba9e3d8ffe9006d9d06b4",
  "data_file": "data/processed/bike_daily.parquet",
  "data_shape": {
    "n_rows": 43338,
    "n_series": 113,
    "date_range_start": "2025-09-25T00:00:00",
    "date_range_end": "2026-10-24T00:00:00"
  }
}
```

Le script crée un dossier distinct sous `data/results/` pour chaque exécution.
Il y sauvegarde les prévisions, les scores et le JSON, avec un même préfixe
horodaté dans leurs noms. Les chemins effectifs enregistrés par le script sont
absolus ; ceux de l'exemple sont abrégés pour la lisibilité. La liste de
distributions et l'empreinte du fichier de verrouillage sont également
illustratives.

Le dossier est créé avant l'écriture des fichiers : une erreur pendant la
sauvegarde peut donc laisser un dossier incomplet.

---

## ✅ État et prochaines étapes

**En place :**
- Le JSON est sauvegardé aux côtés des résultats et contient le contenu YAML,
  un commit abrégé, la branche, l'état Git et des statistiques sur les données.
- Les empreintes SHA-256 des données et de `uv.lock`, la version de Python et
  les distributions Python visibles sont enregistrées.

**À compléter pour une meilleure reproductibilité :**
- Conserver ou retrouver les fichiers correspondant aux empreintes ; une
  empreinte seule ne contient ni les données ni le fichier de verrouillage.
- Tenir compte du système et des dépendances natives : l'inventaire Python et
  l'empreinte de `uv.lock` ne décrivent pas tout l'environnement et ne prouvent
  pas que les distributions installées correspondent au fichier verrouillé.
- Pour un résultat de référence, utiliser un état de code committé et vérifier
  `git_dirty`, en tenant compte des fichiers ignorés.
- Vérifier qu'une nouvelle exécution, avec les mêmes entrées et environnement,
  reproduit les résultats. Le JSON actuel ne démontre pas à lui seul ce point.
