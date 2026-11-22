# Protecting a Lightweight SQL Injection Detector From Unseen Obfuscation

Author: Javier Ignacio Trombetta Ghio. COMP902 Advanced Information Technology
Specialised Project, Auckland Institute of Studies, New Zealand, September to December
2026. Supervisor: Abdul Rehman Ahmed.

This repository compares two defences for a lightweight SQL injection detector,
adversarial training and input normalisation, against sqlmap obfuscation families that
the detector has never seen (leave-one-family-out protocol, five seeds, paired
statistical tests). The detector is a classical machine-learning model on character
n-gram features; three such models are trained, tuned on a validation set and compared
(logistic regression, linear SVM and random forest), and the data are split into explicit
train, val and test sets (70/15/15). A second experiment, the weight analysis, opens the
two linear detectors and traces their decisions to groups of character n-grams. The
repository also includes a demonstration web application in which the trained detector
protects a login page.
Everything runs on a standard laptop; no GPU is needed.

## 1. Requirements

| Software | Version |
| --- | --- |
| Python | 3.15 |
| Node.js | 24 LTS or later (run with 26.13) |
| Git | any recent version |

| Python package | Version |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| sqlglot | 30.24.0 |
| statsmodels | 0.15.0 |

| Node.js package | Version |
| --- | --- |
| express | 4.22.3 |
| express-session | 1.19.0 |
| ejs | 3.1.10 |
| better-sqlite3 | 12.11.1 |

All packages, including their dependencies, are pinned in `requirements.txt` and
`demo/package-lock.json`, and are installed in steps 2 and 3 of section 3.

External repositories, not included here and cloned next to this one in step 1 of
section 3:

- sqlmap: https://github.com/sqlmapproject/sqlmap (only its tamper scripts are used)
- HttpParamsDataset: https://github.com/Morzeux/HttpParamsDataset

## 2. Project structure

The layout below is the working folder after step 1 of section 3. Only
`sqli-detector` is this repository; `HttpParamsDataset` and `sqlmap` are separate
repositories that you clone next to it in that step, because the code looks for them
there.

```
comp902/                           working folder (any name)
  HttpParamsDataset/               dataset (cloned in step 1 of section 3)
  sqlmap/                          sqlmap tamper scripts (cloned in step 1 of section 3)
  sqli-detector/                   this repository
    README.md                      this file
    LICENSE                        MIT licence
    requirements.txt               exact versions of every Python package
    .gitignore                     files kept out of version control
    data/                          the three sets written by split_dataset.py, each in its
                                   own directory
      LICENSE                      licence of HttpParamsDataset, from which the sets come
      train/train.csv              70%: fits the models
      val/val.csv                  15%: threshold selection and hyperparameter tuning
      test/test.csv                15%: final evaluation only
    src/
      config.py                    every setting of the experiment (not run directly):
                                   seeds, split proportions, models, hyperparameter grid,
                                   thresholds, n-gram range, sample sizes, augmentation
                                   size, bootstrap resamples and dataset specification
      common.py                    shared building blocks (not run directly): locating
                                   sqlmap and the dataset, loading the data, the eight
                                   tamper families, the normaliser, training and scoring
                                   the detector, and choosing the threshold at a 1%
                                   false-positive rate
      split_dataset.py             step 5: splits the data into train, val and test
      tune_models.py               step 6: chooses each model's hyperparameters on val
      train_models.py              step 7: trains and compares the three models (O1)
      export_model.py              step 8: copies the primary model into the demo
      screening.py                 step 9: how much each tamper family fools each model (O2)
      validity_check.py            step 10: whether tampered payloads remain attacks
      experiment.py                steps 11 and 12: the four defences on held-out families,
                                   with harmless disguised values as a control
      analysis.py                  step 13: summary tables and statistical tests (O4)
      weight_analysis.py           step 14: what the weights of the linear detectors
                                   respond to (weight analysis, O5)
      compare_weight_analysis.py   compares the tables of step 14 with those of another run
      split_analysis.py            supporting analysis: the dataset before the split, and
                                   whether the split is biased
      seed_analysis.py             supporting analysis: whether the results depend on the
                                   seeds
      attribute_weights.py         supporting analysis: the columns of the dataset, the
                                   weighting of the features and the starting weights
      tests/
        test_common.py             19 unit tests of the pipeline
        test_weight_analysis.py    18 unit tests of the weight analysis
    demo/
      app.js                       creates and starts the Express application
      package.json                 Node.js dependencies and npm scripts
      package-lock.json            exact versions of every Node.js package
      config/
        index.js                   port, protection mode, session and time zone
        project.js                 site name and the figures of the study shown on the pages
      models/
        database.js                in-memory SQLite database with three fictional accounts
        userModel.js               account look-up: a parameterised query, and the
                                   deliberately vulnerable findByCredentialsUnsafe,
                                   used only in vulnerable mode
        incidentLog.js             requests blocked since the server started (in memory)
      controllers/
        homeController.js          home page and the form that scores any text
        authController.js          sign-in and sign-out; sessions
        consoleController.js       console shown after signing in
        errorController.js         "page not found" and error pages
      routes/
        index.js                   URL routes; adds the SQL injection guard to the
                                   sign-in when protection is on
      middleware/
        sqliGuard.js               scores the username and password with the detector
                                   and blocks the request (HTTP 403) above the threshold
        requireAuth.js             restricts the console to signed-in users
        securityHeaders.js         security-related HTTP response headers
        requestLogger.js           one log line per request
        viewLocals.js              values shared by every page
      helpers/
        text.js                    converts form values to single strings and marks the
                                   spaces of an n-gram for display
        time.js                    formats the time shown on the console
      security/
        detector.js                JavaScript implementation of the trained detector
        model.json                 exported model (created in step 8; not kept in
                                   version control)
      views/
        home.ejs                   home page
        auth/login.ejs             sign-in page
        auth/blocked.ejs           page shown when a request is blocked
        console/index.ejs          console
        errors/404.ejs             "page not found" page
        errors/error.ejs           generic error page
        partials/                  parts shared by the pages: head, foot, header, footer,
                                   logo, icon and protection note
      public/
        css/main.css               stylesheet
        js/reveal.js               shows or hides the password typed on the sign-in page
        img/lucide/                the eight icons of the site (logo, browser tab icon,
                                   console and sign-in icons): unmodified Lucide files and
                                   their LICENSE
      tests/
        app.test.js                end-to-end tests of both modes and the public pages
        parity.test.js             checks that JavaScript and Python scores match
        parity_cases.json          parity cases (created in step 8; not kept in
                                   version control)
    results/                       pipeline outputs, described in section 4
```

## 3. Running the offline pipeline

Commands are for Windows PowerShell and start in the working folder (`comp902`). On
macOS or Linux, use `python3 -m venv venv` and `source venv/bin/activate`, and write
paths with `/`.

**Step 1. Get the external repositories.** Place this repository (`sqli-detector`) in
the working folder, then clone sqlmap and the dataset into the same folder. The three
folders then sit side by side, as shown in section 2.

```powershell
git clone https://github.com/sqlmapproject/sqlmap.git
git clone https://github.com/Morzeux/HttpParamsDataset.git
```

**Step 2. Create the Python environment.** An isolated environment with the exact
package versions. Activate it again in every new terminal.

```powershell
cd sqli-detector
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Step 3. Install the demonstration packages.**

```powershell
cd demo
npm install
cd ..\src
```

**Step 4. Check the set-up.** The first command confirms that the eight tamper families
are found; the second runs the unit tests and must end with `OK` (37 tests).

```powershell
python -c "import common; print(len(common.load_tampers()), 'families')"
python -m unittest discover tests
```

**The pipeline at a glance.** Steps 5 to 14, each answering one question. O1 to O5 and
RQ1 to RQ4 are the objectives and the research questions of the project report.

| Step | Script | Question it answers | Writes |
| --- | --- | --- | --- |
| 5 | `split_dataset.py` | How is the data divided? | `data/train`, `data/val`, `data/test` |
| 6 | `tune_models.py` | Which hyperparameters work best for each model? | `model_selection.csv` |
| 7 | `train_models.py` | How good is each model on normal data, and did it learn in a right way? (O1) | `model_comparison.csv`, `feature_weights.csv` |
| 8 | `export_model.py` | Does the website run exactly the same model? (O1) | the model and its test cases, in `demo/` |
| 9 | `screening.py` | How much does each tamper family fool each model? (O2) | `screening.csv` |
| 10 | `validity_check.py` | Do the disguised payloads still work as attacks? | `validity.csv` |
| 11-12 | `experiment.py` | Which defence works against an obfuscation never seen in training, and does it tell a disguised attack from a harmless value disguised the same way? (O3) | `experiment.csv` |
| 13 | `analysis.py` | Which differences are real, and how large? (O4, RQ1 to RQ3) | `tests.csv`; prints the tables |
| 14 | `weight_analysis.py` | What do the weights of the linear detectors respond to, and how do the defences change them? (O5, RQ4) | `weight_analysis.csv` and three `weight_analysis_*.csv` |

Three supporting analyses, described after step 14, give the evidence for the split, the
seeds and the features. Every file in `results/` is described in section 4.

**Step 5. Split the data into train, val and test.** Joins the dataset's two files and
divides them, once and always the same way, into three separate sets. Train (70%) is
what the models learn from; val (15%) is used to choose the threshold and the
hyperparameters; test (15%) is used only for the final numbers. Only normal and SQL
injection payloads are kept (30,156 rows), duplicates are removed (there are none), each
set keeps the same share of attacks, and a check confirms that no payload is in two sets.
Reads the dataset; writes `data/train/train.csv`, `data/val/val.csv`, `data/test/test.csv`.

```powershell
python split_dataset.py
```

**Step 6. Choose the hyperparameters on val.** Hyperparameters are settings fixed before
training, such as the regularisation strength `C` or the number of trees. For each model,
this step tries every combination in `config.PARAM_GRID`, trained on train and measured
on val (share of attacks detected at 1% false positives, with F1 as tie-breaker; remaining
ties go to the simplest combination), and keeps the best. The test set is never used.
Writes `results/model_selection.csv`, with the chosen combination marked
`selected = True`. Takes about a minute.

```powershell
python tune_models.py
```

**Step 7. Train and compare the three models (O1).** Trains each model with its chosen
hyperparameters, sets its threshold at 1% false positives on val, and evaluates it once
on test. It prints the F1 of each model on train, val and test at the same threshold:
similar values mean the model learnt in a right way, while a train value much higher
than val and test would mean overfitting. It also lists the n-grams that weigh most in
each model, evaluates the primary model on the train/test split distributed with the
dataset (row `official`), and checks O1 (F1 of at least 0.95).
It must end with `O1 MET`. Writes `results/model_comparison.csv` and
`results/feature_weights.csv`.

```powershell
python train_models.py
```

**Step 8. Copy the model into the demonstration.** Trains the primary model again,
exactly as step 7 does, which gives the same model, and stops if its threshold is not the
one step 7 stored in `results/model_comparison.csv`. It then writes the model as numbers
the website's JavaScript can read, plus 303 texts scored by Python, which the website's
tests use to check that it scores exactly the same. Writes `demo/security/model.json` and
`demo/tests/parity_cases.json`.

```powershell
python export_model.py
```

**Step 9. Screen the tamper families (O2).** Disguises every test attack with each of the
eight families, once, and measures how many disguised attacks each model still catches,
and how far detection drops from the undisguised attacks, at 1% false positives and at
0.5. The three models are scored on exactly the same disguised attacks. This is a
single-training view; the five-seed answer to RQ1 is column A of the main table printed in
step 13. Writes `results/screening.csv`.

```powershell
python screening.py
```

The disguised attacks are the same 13,024 payloads every time, because the families are
applied with seed 0 and nothing is trained to produce them. They are therefore not stored
by default; `python screening.py --variants` also writes them to `results/variants.csv`,
each test attack next to its disguised version.

**Step 10. Check that the disguised attacks still work.** For 50 test attacks per
family, checks whether the disguised payload still parses as SQL and whether it reaches
the database as the original attack after the single URL decoding performed by Express
(checked by a demo test). A family that evades the detector but breaks the attack is
reported as detector evasion only. Writes `results/validity.csv`.

```powershell
python validity_check.py
```

**Step 11. Run the main experiment for each model (O3).** For each disguise, trains four
versions of a model without that disguise and then attacks them with it: A, no defence;
B, adversarial training with the other seven families; C, input normalisation; D, both.
B and D use the same training variants, so D differs from B only by normalisation. Each
version gets its own threshold at 1% false positives on val. As a control, each version
also scores the normal test values disguised by the same families: a version that flags
everything that is encoded would otherwise look like a good defence. It is run for the
three models and five seeds; the seed changes the sample used for adversarial training,
the randomised families and the random forest. Every run appends to
`results/experiment.csv`; re-running a seed replaces its rows in the analysis.

```powershell
foreach ($m in "logreg", "linsvc", "rf") { foreach ($s in 0..4) { python experiment.py $s --model $m } }
```

The same, one command at a time, is `python experiment.py <seed> --model <model>` for
seeds 0 to 4 and models `logreg`, `linsvc` and `rf`. A seed takes a few minutes with
`logreg` or `linsvc` and several times longer with `rf`, which uses all available cores.

**Step 12. Repeat it leaving out whole groups.** The same, but leaving out a whole group
of families at once (the five encoding families, or the three rewriting families), to
check that a held-out family is not simply similar to one seen in training. Results go
to the same `results/experiment.csv`, marked `protocol = group`.

```powershell
foreach ($m in "logreg", "linsvc", "rf") { foreach ($s in 0..4) { python experiment.py $s --groups --model $m } }
```

**Step 13. Analyse the results (O4).** Turns the raw runs into the tables of the report,
for all models at once. It compares the versions pairwise (B vs A, C vs A, D vs B,
C vs B) with McNemar's exact test and Holm's correction within each seed, and reports
how large each difference is, with a bootstrap 95% confidence interval and the odds
ratio. The summary separates the families the normaliser reverts by construction and
those that reach the backend as attacks, and puts the three models side by side.
Writes `results/tests.csv`. The main table (detection of each version as mean ± SD
over the seeds, at the 1% threshold and at 0.5), the summary per model and version, and
the control table are printed: they are computed from `results/experiment.csv` and
`results/validity.csv`, so running the command again gives them back in seconds.

```powershell
python analysis.py
```

**Step 14. Open the linear detectors: the weight analysis (O5).** Rebuilds versions A, B
and D of the main experiment for the unseen family base64encode, with the same variants,
the same random state and the same threshold rule, and checks them against
`results/experiment.csv`. It then splits every decision into three parts with no
remainder: the baseline (intercept minus threshold), the attack evidence (n-grams with
positive weights) and the normal evidence (n-grams with negative weights), in total and by
eight groups of n-grams. Part 1 reads the undefended detector on plain and encoded
inputs; Part 2 trains the detector with each training family alone and without each
family, to find where the gain of adversarial training comes from; Part 3 goes from B to D
in three steps; Part 4 repeats A, B and D for every value of `C` in `config.PARAM_GRID`.
The paired tests are those of step 13, and the decision rules fixed before the run are
applied by the script. It is run for the two linear models; the random forest has no
weight per n-gram. Each model takes about seven minutes on two cores and must end with
`WEIGHT ANALYSIS CHECK OK`. Writes `results/weight_analysis.csv` and three
`results/weight_analysis_*.csv`.

```powershell
python weight_analysis.py
python weight_analysis.py --model linsvc
```

`python weight_analysis.py --summary` (with `--model linsvc` for the second model) prints
every table again from the stored files, with the means over the seeds, the split of every
gap and the verdict of each decision rule; none of these is stored, because they are
computed from the stored files. `python compare_weight_analysis.py --reference <folder>`
compares the four tables with those of a run made on another machine and ends with
`All files match the reference` when nothing differs.

**Supporting analyses.** Three scripts give the evidence for choices made in the main
experiment. They are run after step 13 and none of them changes a file of the pipeline.

```powershell
python split_analysis.py
python split_analysis.py --model linsvc
python split_analysis.py --model rf
foreach ($m in "logreg", "linsvc", "rf") { python seed_analysis.py --model $m }
python attribute_weights.py
python attribute_weights.py --model linsvc
python attribute_weights.py --model rf
```

- `split_analysis.py` describes the dataset before it is split, tests whether train, val
  and test are alike, measures how the detector performs on test rows that resemble the
  training set and on those that do not, removes one kind of SQL injection from training
  to test it unseen, and repeats the split with every seed in two ways. It reports two
  threshold rules side by side: the one used in this study (exact 1% false positives on
  val) and the one used before it was corrected, which read only the corner points of the
  ROC curve. Writes `results/split_profile.csv`, `results/split_similarity.csv`,
  `results/split_unseen_kind.csv` and `results/split_repeats.csv`.
- `seed_analysis.py` repeats the part of the experiment where the versions differ
  (base64encode and the encoding group withheld) with three sets of five seeds, and
  varies each source of randomness alone. It ends by comparing the seeds of the study
  with `results/experiment.csv`. `python seed_analysis.py --summary` prints the tables.
  Writes `results/seed_analysis.csv`.
- `attribute_weights.py` measures what each column of the dataset adds, compares five
  ways of weighting the same n-grams, class weights, and, for logistic regression, the
  starting weights of the training. Each variant is also scored on harmless values that
  are encoded or long. Writes `results/attribute_weights.csv` and
  `results/attribute_columns.csv`.

## 4. Outputs

All outputs are written to `results/`. A file is stored only when two things hold: its
figures were measured, so the other files cannot give them back without training the
models again; and its rows describe something that no other file describes. Tables that
are computed from the stored files are printed by the script that computes them, and
measurements of the same thing are columns of the same file. This leaves eighteen files.

**The main experiment (steps 6 to 13): seven files.**

| File | One row per | Content | Answers |
| --- | --- | --- | --- |
| `model_selection.csv` | model and hyperparameter combination | Every combination tried on val; the chosen one is marked `selected = True` (step 6) | How the hyperparameters were optimised |
| `model_comparison.csv` | model and split | The three models on train, val and test, plus the primary model on the train/test split distributed with the dataset (step 7) | O1; the model comparison; overfitting check |
| `feature_weights.csv` | model and n-gram | For each linear model, the 20 n-grams that weigh most towards attack and the 20 that weigh most towards normal; for the random forest, the 20 most important (step 7) | Which features weigh most |
| `screening.csv` | model and tamper family | Effect of each tamper family on each model, single training (step 9) | O2; first view of RQ1 |
| `validity.csv` | tamper family, plus one row for the untampered attacks | Whether tampered payloads remain valid SQL and reach the backend as attacks (step 10) | Which families still work as attacks |
| `experiment.csv` | model, protocol, held-out family or group, seed and version | Every run of steps 11 and 12, with its control: 600 rows when complete (3 models x 5 seeds x (8 families + 2 groups) x 4 versions) | O3 (raw data); RQ1 to RQ3 |
| `tests.csv` | model, protocol, held-out family or group and comparison | Paired comparisons B vs A, C vs A, D vs B and C vs B: 120 rows when complete (step 13) | O4 |

**The weight analysis (step 14): four files.**

| File | One row per | Content | Answers |
| --- | --- | --- | --- |
| `weight_analysis.csv` | model, seed, part, version, value of `C` and input set | Every measurement: the share flagged, the mean margin, the baseline, the attack and normal evidence in total and for each of the eight n-gram groups, the coverage and the AUC | O5 (raw data); RQ4 |
| `weight_analysis_ngrams.csv` | model, seed, comparison and n-gram | The 20 n-grams with the largest difference in contribution for each gap, and one payload worked through n-gram by n-gram, plain and in base64 | Which n-grams carry each difference; check of the grouping |
| `weight_analysis_facts.csv` | subject and measure (model and seed where they apply) | What each class of the data contains (upper-case letters, percent signs, URL escapes, equals signs, digits, whitespace), what the normaliser does to the training set, the variants and base64 text, and where upper-case letters enter the training set | Premises of the answers of the weight analysis |
| `weight_analysis_tests.csv` | model, part and comparison | Paired comparisons of Parts 2, 3 and 4, with the tests of step 13 | Which differences are real |

**Supporting analyses: seven files.**

| File | One row per | Content | Answers |
| --- | --- | --- | --- |
| `split_profile.csv` | characteristic of the data | What the dataset contains before the split (rows, classes, lengths, kinds of value, near-duplicates, symbols) and, for the characteristics compared between sets, their value in train, val and test with the test of whether the sets differ | Is the split balanced? |
| `split_similarity.csv` | model, threshold rule, class and group of test rows | How close the test rows are to the training set, and how the detector performs on each group | Is the test set too close to the training set? |
| `split_unseen_kind.csv` | model, threshold rule and kind of SQL injection | Detection of one kind of SQL injection when it is removed from training | Does the detector generalise to a kind of attack it never saw? |
| `split_repeats.csv` | model, way of splitting, seed of the split and threshold rule | The split repeated with every seed, in two ways, and the detector measured each time | Do the results depend on one particular split? |
| `seed_analysis.csv` | model, held-out unit, the three seeds and version | The versions trained with other seeds, and with each source of randomness varied alone | Do the results depend on the seeds? |
| `attribute_weights.csv` | model, part and variant of the detector | Each variant on the clean test set, on the tampered attacks and on harmless values that are encoded or long | Which columns, which weighting, which starting weights |
| `attribute_columns.csv` | column of the dataset | What each column holds, its role and the check made on it | Which columns are used |

### What is printed and not stored

| Table | Printed by | Computed from |
| --- | --- | --- |
| Main table: detection of each version as mean ± SD over the seeds, per model, threshold, protocol and held-out family or group (RQ1 in column A, RQ2, RQ3) | `python analysis.py` | `experiment.csv` |
| Summary per model and version: mean detection over subsets of families and on each held-out group, clean F1 and false-positive rate | `python analysis.py` | `experiment.csv`, `validity.csv` |
| Control table: attacks detected next to harmless disguised values flagged | `python analysis.py` | `experiment.csv` |
| Means over seeds, the split of every gap and the verdicts of the decision rules of the weight analysis | `python weight_analysis.py --summary` | `weight_analysis.csv`, `weight_analysis_tests.csv` |
| Tables over the seed sets | `python seed_analysis.py --summary` | `seed_analysis.csv` |

### How to read the columns

Three rules apply to every file:

- **Operating point.** Unless stated otherwise, every figure is measured at the threshold
  that keeps the false-positive rate at 1% on val. A suffix `_t05` means the default
  threshold of 0.5 instead.
- **Units in the name.** `_pct` is a percentage (0 to 100), `_pp` a difference in
  percentage points; F1 is between 0 and 1.
- **Conditions.** A, no defence; B, adversarial training; C, input normalisation; D, both.

| File | Column | Meaning |
| --- | --- | --- |
| `model_selection.csv` | `params` | Hyperparameters tried (JSON) |
| | `threshold` | Score threshold that gives a 1% false-positive rate on val |
| | `val_detection_pct`, `val_f1` | Attacks detected and F1 on val at that threshold |
| | `selected` | `True` for the combination used from then on |
| `model_comparison.csv` | `split` | `70/15/15` (this study) or `official` (the train/test split distributed with the dataset, primary model only) |
| | `params`, `threshold` | As above |
| | `f1_train`, `f1_val`, `f1_test` | F1 on each set at the threshold; similar values mean no overfitting |
| | `f1_test_t05` | F1 on test at the threshold of 0.5 |
| | `detection_test_pct`, `fpr_test_pct` | Attacks detected and normal payloads wrongly blocked, on test |
| `feature_weights.csv` | `measure` | `weight` (linear models) or `importance` (random forest) |
| | `direction` | `attack` or `normal` for weights; `unsigned` for importances |
| | `rank`, `ngram`, `value` | Position, character n-gram, and its weight or importance |
| `screening.csv` | `modified_pct` | Share of test attacks that the family actually changes |
| | `detection_pct`, `detection_t05_pct` | Tampered attacks detected |
| | `drop_pp`, `drop_t05_pp` | Fall from the detection of the untampered attacks; negative means a slight rise |
| `validity.csv` | `syntactically_valid_pct` | Tampered payloads that still parse as SQL |
| | `reaches_backend_pct` | Tampered payloads that reach the database as the original attack after one URL decoding |
| `experiment.csv` | `protocol`, `held_out` | `family` or `group`, and the family or group left out of training and used to attack |
| | `seed`, `condition` | Seed (0 to 4) and condition (A to D) |
| | `detection_pct`, `detection_t05_pct` | Held-out attacks detected |
| | `clean_f1`, `clean_fpr_pct` | F1 and false-positive rate on the untampered test set |
| | `normal_held_out_flagged_pct` | Control: normal test values disguised by the held-out family or group, flagged |
| | `auc_held_out` | Control: how well the scores separate held-out disguised attacks from normal values disguised the same way (1 = perfectly, 0.5 = not at all); it does not depend on the threshold |
| | `normal_urlquote_flagged_pct`, `urlquote_changed_pct` | Control: normal test values percent-encoded the standard way that are flagged, and the share of normal values that this encoding changes |
| | `normal_<family>` | Control: normal test values disguised by each of the eight families, flagged (in %) |
| `tests.csv` | `comparison` | `new vs reference`, for example `D vs B` |
| | `difference_pp` | Mean difference in detection, new minus reference |
| | `ci95_low_pp`, `ci95_high_pp` | Bootstrap 95% confidence interval of that difference |
| | `odds_ratio` | Attacks only the new condition caught, divided by attacks only the reference caught, pooled over seeds; `inf` if the new lost none, `0` if it caught none, empty if both agreed on every payload |
| | `p_holm_median` | Median over seeds of the Holm-corrected McNemar p-value |
| | `significant_seeds` | Seeds in which the difference is significant after Holm correction, for example `5/5` |
| `weight_analysis.csv` | `part`, `condition`, `C`, `input_set` | Part of the analysis (1 to 4), the detector (A, B, D or a variant of them), the value of `C` in Part 4, and the inputs scored |
| | `flagged_pct`, `mean_margin` | Share of the inputs flagged, and their mean distance to the threshold (positive = flagged) |
| | `baseline`, `attack_evidence`, `normal_evidence` | The three parts of the mean margin; they add up to it |
| | `attack_<group>`, `normal_<group>` | The attack and the normal evidence divided among the eight groups of n-grams |
| | `coverage_pct`, `auc`, `clean_fpr_pct` | Share of the n-grams of the inputs that the detector knows; separation of the attacks from the normal values of the same input set; false positives on the plain test set |
| | `intercept`, `threshold`, `tau` | Intercept of the model; its score threshold; and the same threshold on the scale of the margin |
| | `boundary_weight`, `residual` | Weight of the word-boundary n-gram, which every input has; largest difference between the sum of the parts and the model's own decision value |
| | `n`, `vocabulary`, `train_rows` | Number of inputs scored, n-grams the detector knows, and rows it was trained on |
| `weight_analysis_ngrams.csv` | `comparison`, `ngram`, `group`, `rank` | The gap, the n-gram (between square brackets, so that boundary spaces stay visible), its group and its position |
| | `weight_x`, `weight_y`, `tfidf_x`, `tfidf_y`, `contribution_x`, `contribution_y`, `difference` | Weight, mean TF-IDF value and mean contribution of the n-gram in the two detectors compared, and the difference between the contributions |
| `weight_analysis_facts.csv` | `subject`, `measure`, `value` | What is described, what is measured, and the figure |
| `weight_analysis_tests.csv` | | The columns of `tests.csv`, with the part of the analysis instead of the protocol |
| `split_profile.csv` | `section`, `item` | The characteristic and the category or statistic within it |
| | `value`, `share_pct` | Count (or length in characters) in the whole dataset, and its share |
| | `train`, `val`, `test`, `unit` | The same characteristic in each set, in the unit given |
| | `p_value`, `p_train_vs_val`, `p_train_vs_test`, `present_in_all_sets` | Test of whether the sets differ (chi-square for categories; Kolmogorov-Smirnov for lengths, train against each other set), and whether the category occurs in the three sets |
| `split_similarity.csv` | `threshold_rule`, `threshold` | `exact 1% FPR` (this study) or `ROC corners (old rule)`, and the threshold it gives |
| | `class`, `group`, `rows`, `share_pct` | Class, group of test rows by closeness to the training set, and its size |
| | `median_similarity`, `flagged_as_attack_pct` | Median cosine similarity to the nearest training row of the same class, and share of the group flagged |
| `split_unseen_kind.csv` | `kind_removed_from_training`, `training_attacks_removed`, `test_attacks_of_that_kind` | The kind of SQL injection withheld and the rows involved |
| | `detected_when_seen_pct`, `detected_when_unseen_pct`, `false_positive_rate_pct` | Detection of that kind with the normal split and when withheld, and the false-positive rate of the detector trained without it |
| `split_repeats.csv` | `strategy`, `split_seed` | How the split is stratified, and its seed |
| | `threshold_rule`, `threshold` | As in `split_similarity.csv` |
| | `val_fpr_pct`, `clean_f1`, `clean_fpr_pct`, `clean_detection_pct` | False-positive rate on val at the threshold, and the clean test results |
| | the eight family names, `encoding_mean_pct`, `rewriting_mean_pct` | Tampered test attacks detected per family with no defence, and the means of the two groups |
| `seed_analysis.csv` | `sample_seed`, `tamper_seed`, `model_seed` | The seed of each source of randomness: the sample for adversarial training, the randomised families and the random forest |
| | `detection_pct`, `clean_f1`, `clean_fpr_pct` | As in `experiment.csv` |
| `attribute_weights.csv` | `part`, `variant` | `1 columns`, `2 weighting`, `3 class weights` or `4 starting weights`, and the variant of the detector |
| | `clean_f1`, `clean_fpr_pct`, `clean_detection_pct` | The variant on the clean test set |
| | the eight family names, `encoding_mean_pct`, `rewriting_mean_pct` | Tampered test attacks detected per family, and the means of the two groups |
| | `normal_<family>`, `normal_encoded_flagged_pct` | Normal test values under each encoding family that are flagged, and their mean |
| | `normal_long_flagged_pct`, `normal_long_nosep_flagged_pct`, `normal_long_comma_flagged_pct`, `normal_repeated_flagged_pct` | Long harmless inputs flagged: eight normal values joined by a space, by nothing or by a comma, and one value repeated six times |
| | `iterations`, `largest_weight`, `max_diff_from_zero_start` | Part 4 only: iterations of the training, the largest weight, and the largest difference from the weights reached from a zero start. The rows of part 4 fill only these columns, `clean_f1` and `encoding_mean_pct` |
| `attribute_columns.csv` | `column`, `content`, `role`, `check` | Column of the dataset, what it holds, how it is used, and the check made on it |

The folder `results/predictions/` holds working files that are not kept in version
control: `experiment_<model>_<seed>.npz` (steps 11 and 12) and `weight_<model>_<seed>.npz`
(step 14), one per model and seed. Each holds, for every condition, an array with one
entry per test attack in a fixed order: 1 if the condition detects it at its 1%
false-positive threshold and 0 otherwise. The paired tests of steps 13 and 14 read them;
without them `analysis.py` still prints the tables but skips the tests.

## 5. Models, features and hyperparameters

**Features.** Every payload is turned into character n-grams of 1 to 4 characters
inside word boundaries (scikit-learn's `char_wb` analyser), weighted by TF-IDF, with
lower-casing disabled so that the randomcase family is not undone by preprocessing, and
keeping n-grams that appear in at least two training payloads.
On the training set this gives about 52,000 features. Character n-grams capture patterns
inside tokens, such as quotes, operators and fragments of SQL keywords, rather than whole
words. This motivated the choice; it is a design decision of this study, not a result
it tests.

**Models.** Three classical machine-learning models learn from the same features. All
train on a CPU in seconds to minutes; deep learning is out of scope because the dataset
is small (30,156 rows) and no GPU is available.

| Model | What it learns | Hyperparameters tuned | Role |
| --- | --- | --- | --- |
| `logreg` | Logistic regression with L2 regularisation: one weight per n-gram; the score is the logistic function of the weighted sum | `C` (inverse regularisation strength) | Primary model: exported to the demonstration (its weights are a plain list of numbers, easy to run in JavaScript) and the default of every script; the experiment runs on all three |
| `linsvc` | Linear support vector machine: a separating hyperplane in the n-gram space; its margin is mapped to (0, 1) with the logistic function so that thresholds are comparable | `C` | Second linear model, with a different training objective |
| `rf` | Random forest: an ensemble of decision trees on random subsets of the n-grams | `n_estimators`, `max_depth` | Non-linear model, for contrast with the two linear ones |

**Weights.** In the linear models, every n-gram has a learnt weight: positive weights
push a payload towards "attack", negative ones towards "normal", and the intercept sets
the starting point. The exported `model.json` contains exactly these weights, the IDF
values and the vocabulary. The random forest has no single weight per feature; its
feature importances play a similar role. `results/feature_weights.csv` lists, for each
linear model, the 20 n-grams that weigh most in each direction and, for the random
forest, the 20 most important. For logistic regression, quotes, parentheses, commas,
hyphens and equals signs push towards "attack", and fragments of ordinary words push
towards "normal".

**Did the models learn in a right way?** Step 7 prints the F1 of each model at the same
threshold on train, val and test. Similar values mean the model generalises; a train
value much higher than val and test would mean overfitting. Val is slightly optimistic,
because the threshold and the hyperparameters were chosen on it; test is the unbiased
figure.

**Hyperparameters and how they are optimised.** `tune_models.py` searches the grid in
`config.PARAM_GRID` for each model, fitting on train and scoring on val only: the
share of val attacks detected at the threshold that keeps the val false-positive rate
at 1%, with F1 as tie-breaker, and the simplest combination winning any remaining tie.
The chosen values are marked `selected = True` in `results/model_selection.csv` and used
everywhere afterwards. The detection threshold itself is also set on val, per model and
per condition. The test set is used once, for the final numbers.

**Data.** The source dataset has 31,067 rows (20,712 in its training file and 10,355 in its
test file), of which 30,156 are normal or SQL injection payloads; the rest (cross-site
scripting, path traversal and command injection) are out of scope. `split_dataset.py`
pools the two files, keeps those 30,156 rows, removes duplicate payloads (there are
none, and the count is printed), and divides the rows 70/15/15 into
`data/train/train.csv`, `data/val/val.csv` and `data/test/test.csv`, stratified by label
and with a fixed seed, so the three sets are explicit, disjoint and reproducible. A unit
test checks that no payload appears in two sets.

**What the five seeds vary, and what they do not.** The data split is fixed, so a seed
changes only the sample of training attacks used for adversarial training, the two
randomised tamper families (randomcase, randomcomments) and the random forest. For the
two linear models, conditions A and C are therefore identical across seeds on the
deterministic families, and their standard deviation is zero by construction. The
per-seed significance counts in `tests.csv` are meaningful for comparisons that involve
B or D; for C vs A on a deterministic family they repeat one test five times and should
be read as a single test. Re-splitting train and val per seed would restore that
variation at the cost of a single explicit split; this study keeps the explicit split.

**Limitations.** A single dataset; a non-adaptive attacker (fixed tamper families); the
detector evaluated on parameter values as sent by the client, before server-side
decoding; and a small hyperparameter grid, chosen to keep every run within minutes on a
laptop.

## 6. Demonstration web application

A small website built with Express that presents the project and has a console behind a
sign-in page. In vulnerable mode, the sign-in builds its database query by string
concatenation; in protected mode, the trained detector scores the username and the
password before the query, which is also parameterised. The home page also has a form
that sends any text to the detector and shows its score. The database lives in memory and
is rebuilt on every start, the accounts are fictional, and the application is for local
use only.

Complete steps 1 to 8 of section 3 first. Then, from `sqli-detector`:

**Step 1. Run the automated tests.** Must end with `pass 32` and `fail 0`. Besides the
pages, sign-ins and sessions, they check that the website scores exactly like the Python
model and that it URL-decodes form values once, which is what step 10 assumes.

```powershell
cd demo
npm test
```

**Step 2. Start the website without protection.** Leave the terminal open while you use
the site; it prints `login protection: OFF`.

```powershell
node app.js --vulnerable
```

**Step 3. Attack the sign-in.** Open http://localhost:3000, select **Sign in**, and
enter the username `' OR 1=1 --` with any password. The attacker reaches the console
without a valid password, and the terminal prints the injected query.

**Step 4. Restart with protection.** Stop the server with `Ctrl + C` and start it again
in protected mode; it prints `login protection: ON`.

```powershell
node app.js
```

**Step 5. Repeat the attack.** The **Request blocked** page appears with the field, the
score, the threshold and an incident reference, which is also logged in the terminal; the
query never reaches the database.

**Step 6. Sign in as a legitimate user.** Use `researcher` / `researcher123` (or
`analyst` / `analyst123`). The console opens normally and lists the request blocked in
step 5. The third account, `operator`, has a password that looks like an SQL injection
(`' OR '1'='1`): with protection on, its sign-in is blocked although the credentials are
correct, which shows the cost of a false positive. Stop the server with `Ctrl + C`.

If the terminal reports that port 3000 is already in use, another instance of the
website is still running: stop it with `Ctrl + C` first.

## 7. Licences and attribution

- This code: MIT (see `LICENSE`).
- HttpParamsDataset: MIT (Morzeux), https://github.com/Morzeux/HttpParamsDataset. The
  three sets in `data/` are taken from it; its licence text is in `data/LICENSE`.
- sqlmap tamper scripts: GNU GPL version 2 or later, with the clarifications in its
  LICENSE file, https://github.com/sqlmapproject/sqlmap. sqlmap is not redistributed
  here; the transformed payloads that `screening.py --variants` writes to `results/` are
  data produced with it, not derived code.
- numpy, scipy, scikit-learn, pandas, statsmodels: BSD. sqlglot: MIT. better-sqlite3,
  express, express-session: MIT. EJS: Apache-2.0.
- `demo/security/detector.js`: the function `charWbNgrams` is a translation into
  JavaScript of the `char_wb` analyser of scikit-learn 1.9.1
  (`CountVectorizer._char_wb_ngrams`, copyright 2007-2026 The scikit-learn developers,
  BSD 3-Clause licence), so that the website cuts a text into the same n-grams as the
  Python model. The rest of the file applies the exported weights with the TF-IDF and
  logistic formulas.
- Icons of the demonstration: eight files copied without changes from Lucide 1.77.0 (the
  `lucide-static` package), https://lucide.dev, under the ISC licence. Two of them,
  `check` and `x`, derive from Feather (Cole Bemis) and are under the MIT licence. Both
  licence texts are in `demo/public/img/lucide/LICENSE`. The site contains no other
  image.
