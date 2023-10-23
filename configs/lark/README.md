# Feishu robot

## Config

Put the webhook path of your FeiShu (Lark) robot into `lark.py`, and make the following settings:

```python
lark = 'https://open.feishu.cn/open-apis/bot/v2/hook/XXXX-XXXX-XXXX-XXXX'
```

**Note:** Pay attention to privacy!

For more details about FeiShu robot, please refer to [here](https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot).

## Running command

If you want to use FerShu robot during training and testing, add `-l` or `--lark` in the running command.

Examples:

```
# training script
python tools/train.py ${CONFIG_FILE} -l ${Other setting}

# testing script
python tools/test.py  ${CONFIG_FILE} ${CHECKPOINT} -l ${Other setting}
```
