import sys
import pandas as pd
import pathlib

class InputError(Exception):
    pass

if len(sys.argv) < 3:
    raise InputError("""There must be at least two .csv files to merge.
                        
                        python -m data.merge_csv <file1.csv> <file2.csv> <...> <filen.csv>""")
out = pd.read_csv(sys.argv[1], index_col=0)
for f in sys.argv[2:]:
    df = pd.read_csv(f, index_col=0)
    out = pd.concat([out, df])
    
out = out.fillna(0)

OUTPATH = input("Where should the result file be written? ")

out.to_csv(pathlib.Path(OUTPATH) / "merged_params.csv")
print("Successfully wrote 'merged_params.csv', ending.")
