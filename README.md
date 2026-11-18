<div align="center">

# retrieval-tuner-local

**Grid search de retrieval** · Angel Salazar

`top_k × min_score → hit_rate`

</div>

---

### `$ whoami`

En vez de adivinar hiperparámetros del retriever, los mido en local y me quedo con el mejor combo.

---

### `$ ./tune.sh`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pytest
python -m retrieval_tuner
```

Usa el `best` en `local-rag-pipeline` (`.env`: `TOP_K`, `MIN_SCORE`).

---

Hecho en Lima · Angel Salazar · CIP 404730
