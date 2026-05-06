This is transformer from scratch's first part, tokenisation part. Transformer’s first step is tokenisation. In this model you use BPE (Byte Pair Encoding) tokenisation model. The main reason for using it is that it solves unknown , subwords, words problems which you usually face in other tokenisation models. 

**Pre Tokenisation:**
Pre tokenisation is a step where you split our text by whitespaces to get individual words and count their occurrences. If word “NEW” comes five times in a text, it becomes {’N E W <\w>’:5}. We add <\w> at the end of each word, which represents end of word. 

**Tokenisation:**
Our get_stats and merge_vocab functions are part of tokenisation step. In get_stats function which gets the words and their count in pairs. Like if you have “E A T”: 2. It mean word EAT occurred two times. Now for tokenisation it becomes (E, A):2, (A, T):2, this is what our get_stats function is doing. 
In second function which is merge_vocab, you are combining mostly occurred paired. If you take another example , for word {“N E W E S T”:1}, its pairs would be (N E):1 , (E W):1, (W E):1, (E S):1, (S, T):1. It would merge mostly occurred like NE W E S T: 1. In second iteration it would become (NE, W):1, (W, E):1, (E, S):1 , (S, T):1 so after merge_vocab function {“NEW E S T” :1}. This will go till last iteration and then in last iteration It would become (‘NEWES’ ’T’) -> {‘NEWEST’:1}

As frequency of each pair is same so it will pick first one, otherwise it pick with the highest frequency. 
