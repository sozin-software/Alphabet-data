# Alphabet dictionary data

The offline dictionary that the Alphabet app (by Sozin) downloads on first launch. Each release
holds one compressed SQLite file built from free sources; see LICENSE (CC BY-SA 4.0).

- Build it: Actions → "Build and publish the dictionary" → Run workflow.
- The app downloads `releases/latest/download/alphabet-<format>.db.gz`.
- How the file is made: `scripts/build_dictionary.py`.
