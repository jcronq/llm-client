# LLM Client

This is a client for running experiments.  Remember to update this readme when more abilities are added.

## Experiments

Put all of your experiments in the experiments directy off root.

They're written in .yaml, and are self-explanatory.  Or if they aren't self-explanatory just ask me: @jcronq.

## Run Instructions
copy .env.example to .env
Add your OpenAI API Key to .env 
install dependencies (probably all there, install using poetry)

```bash
poetry init
poetry install
```

Then to run an experiment
```bash
python main.py experiments/<your-experiment>.yaml
```

Results will be stored in `<your-experiment>-results.json` and `<your-experiment>-results.csv`.  The .csv file might be broken after adding labels.  (I like to use commas for lists!)

## Operating cost
I run quite a few experiments.  Costs < $0.02 / day on gpt-3.5-turbo.  Not sure what gpt-4 will cost.



