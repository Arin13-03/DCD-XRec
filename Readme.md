# DCDXRec

## Datasets


From [Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html), download "metadata" and "ratings only" for:
- Movie (Movies and TV) into `datasets/raw/Amazon/Movie`
- Music (CDs and Vinyi) into `datasets/raw/Amazon/Music`
- Cell (Cell Phones and Accessories) into `datasets/raw/Amazon/Cell`
- Elec (Electronics) into `datasets/raw/Amazon/Elec`

### Filtering

```{bash}
cd datasets
```

Then execute filtering command (with setting `--domain` to specify dataset):
```{bash}
python filter.py
```
The script takes inputs of filtered files from `raw` folder and save outputs into `processed` folder.

### Processing

After filtering, execute processing command (with setting `--domains` to specify dual datasets):
```{bash}
python process.py
```