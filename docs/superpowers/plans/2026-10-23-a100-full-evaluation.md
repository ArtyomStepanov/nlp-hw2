# Full BART evaluation and actual T4 timing

User explicitly authorized both remaining checks with «давай делать» after the full-training completion report. The accidental off-topic LLM message was withdrawn. No new training is authorized or needed.

- Finalist: full BART8+8/d1024/FF4096, 244391936 parameters, best step180000, final export bit-identical to best; remote artifacts/a100_full_20261023/training/trial/final. Weight SHA2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165.
- Pass@1: original fixed1M/20k bundle SHA7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1; canonical validation SHA282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406; 19886 eligible and114 excluded. Same evaluator/golden references, beam4/LP1/max256/batch64. Prove all selected validation token rows match full-training validation at saved original indices; do not fake manifests to bypass cross-bundle identity checks.
- T4: actual final weights on agreed Colab notebook, all1919 official questions in original order, fixed decoder beam4/LP1/max256, FP16 AMP, sorting by source length. Initial batch128 follows prior actual Colab recipe; batch is a hardware throughput choice. No test targets or quality-based decoder selection. Timer includes tokenization, transfers, generation and decoding; separately report load/setup and any uncertainty about official timing scope.
- Private transfer via SSH/SCP to local artifact then Colab Files, no public weight hosting or credential transfer. This necessary transfer is covered by the explicit actual-model T4 benchmark request. Preserve old notebook cells/outputs and all immutable training snapshots. No interruption of other active runtimes.
- Ruling: current shared checkout plus a new isolated remote evaluation snapshot; no Git commits, push, reset or worktree migration.

## Execution ledger

- [x] Extend evaluator with explicit verified full-training bundle provenance; default historical evaluator behavior unchanged. Agent inspect_full_training_path owns evaluator and focused tests.
- [x] Freeze isolated evaluation code, launch one evaluation in remote tmux, verify actual progress and all reference identities, download small results.
- [x] Package actual final weights, exact config/tokenizer/code/test and SHA manifest; upload to private Colab filesystem.
- [x] Run actual T4 benchmark, verify GPU/model identity, full row count/order and observed timing; archive metrics and proof screenshot.
- [x] Independently review final comparisons, update README and remaining-step reports. No quality claim from loss alone.

Current T4 tab: nlp_hw2_T4_quick_check.ipynb at https://colab.research.google.com/drive/1xdx4Dil5nVZK_l8FlVZPJC7zLM8kKgiY. Was disconnected with old saved139M outputs; connected to a newly allocated T4 without running old cells. Source benchmark reusable: inference.py + all models/*.py + utils.py. Local official test data/tinysms_qa_12m/data/test/test.parquet, rows1919, SHA8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a, question-order SHA6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279.

- Evaluation launched in isolated26-file snapshot, source archive SHAed8113dc54e9228c83f834025476abfdd44d3553147f7a112bf9d7ea1093956d. Pipeline228251, driver228257; original20k/full118302 mapping passed, first1088 predictions generated normally.13 new +1 old integration tests passed; independent review no blockers.

- T4 archive complete/frozen,978377144bytes SHA9845b2ad033482ad113dfc02592bbf72963520e4cf0c8c83403cd3ff144260da. Browser auto-review initially rejected upload for missing explicitpayload/destinationapproval; human then explicitly approved «Да, загрузить в этот Colab». Private upload running; newcell75vd4gQhDoRa waits for exact bytes, verifiesSHA, safelyextracts then executesactualbenchmarkbatch128.

- Pre-execution review found Transformers5.10.2 GenerationConfig standarddefaults represented asNone. Original frozenbenchmark strictvalidator would rejectbeforetiming. Six focusedtests confirm effective-defaultfallback acceptsactualsavedconfig withoutmutatinggeneration and rejectsnondefaults. Ownwaitingcell interrupted beforemodelload; uploadcontinued. NewColabcell20 createsseparate hardlinkedweightsoverlay withonlyvalidationfallback, updatesmanifestandrepairreceipt, preservesoriginalarchive. Actualtimer remains no-warmup fixeddecoder.

- Pass@1 complete exit0 at2026-10-23T14:07:44UTC:14775/19886=74.29850146%, +1118/+5.62204566pp vsdistinct4m. All15smallreportfileSHA and20krow/golden identities independently verified; paired13101both/1674fullonly/556oldonly/4555neither.
- Actual T4 complete2026-10-23T14:09:14UTC:165.073922592s inner/165.106611568s outer, load4.298147315s, peak8.069822GiB,1919orderednonemptyEOS answers, no truncation or tokenlimit. Under180s in this genuine full-finalist run. Download326792byteZIP SHA c53116eed939b7b4f021826d9cbd46a56ed56eb1da13c67000a99c583c83673c; screenshot t4_proof.png. No external submission.

- Independent final T4 checkpassed: all9receipts +6resultfileSHA, every1919question/orderedcode,15batchcoverage, exactsinglevalidatorpatch andunchangedcode/config/tokenizer, actualbestweightprovenance, synchronizedtimers. load+outer169.404758883<180. No findings. README and completion/remaining-state reports updated; no authorizedmeasurements remain.

- Repository benchmark helper now directly includes the same tested validator-only fallback as the executed T4 overlay (source SHA41343c797a415237bbd98bedbcdf4f27d9ec848988e1d7c0807f08884fd36829). Frozen original archive remains unchanged for provenance.
