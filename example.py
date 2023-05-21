from transformers import BertTokenizerFast, GPT2LMHeadModel, TextGenerationPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = BertTokenizerFast(vocab_file="gpt2model/vocab.txt", sep_token="[SEP]", pad_token="[PAD]",
                                  cls_token="[CLS]")
model = GPT2LMHeadModel.from_pretrained("gpt2model")
text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("[CLS]历 史 名 城 ， 九 水 回 澜 ， 飞 扬 吴 楚 三 千 韵 -",max_length=64, do_sample=True))



prefix = "根据上联写出下联，上联："
input = "一举三思，尽由人心宛转"
prefix2 = "，下联："
tokenizer = T5Tokenizer.from_pretrained("mengzi")
model = T5ForConditionalGeneration.from_pretrained("mengzi")
input_ids = tokenizer(prefix+input+prefix2, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True,max_length = 64))


input = "请给出一副关于喝酒的对联"
tokenizer = T5Tokenizer.from_pretrained("mengzi_generate")
model = T5ForConditionalGeneration.from_pretrained("mengzi_generate")
input_ids = tokenizer(input, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True,max_length = 256))
