This is transformer from scratch. Transformer’s first step is tokenisation. In this model you use BPE (Byte Pair Encoding) tokenisation model. The main reason for using it is that it solves unknown , subwords, words problems which you usually face in other tokenisation models. 

**Pre Tokenisation:**
Pre tokenisation is a step where you split our text by whitespaces to get individual words and count their occurrences. If word “NEW” comes five times in a text, it becomes {’N E W <\w>’:5}. We add <\w> at the end of each word, which represents end of word. 

**Tokenisation:**
Our get_stats and merge_vocab functions are part of tokenisation step. In get_stats function you gets words and their count in pairs. Like if you have “E A T”: 2. It mean word EAT occurred two times. Now for tokenisation it becomes (E, A):2, (A, T):2, this is what get_stats function is doing. 
In second function which is merge_vocab, you are combining mostly occurred paired. If you take another example of word {“N E W E S T”:1}, its pairs would be (N E):1 , (E W):1, (W E):1, (E S):1, (S, T):1. It would merge mostly occurred like NE W E S T: 1. For second iteration it would become (NE, W):1, (W, E):1, (E, S):1 , (S, T):1 so after merge_vocab function {“NEW E S T” :1}. This will go till last iteration and then for last iteration it becomes (‘NEWES’ ’T’) -> {‘NEWEST’:1}

As frequency of each pair is same so it will pick first one, otherwise it picks the word with highest frequency. 

**Positional Embedding:**
Then comes positional embeddings. Before this you generate embedding matrix and initialise it with random values. The standard sinusoidal formula is :

**PE_(pos, 2i) = sin(pos / 10000^2i/d_model)**

**PE_(pos, 2i+1) = cos(pos / 10000^2i/d_model)**

Pos -> it is position of a word, generated previously and it represents row
2i  -> it represents col. We use it in loop so “I” represents iteration. For even positions we use 2i and for odd positions we use 2i+1. 
We use sin and cos function, think of sin and cos as longitude and latitude of word. One represents longitude and second one represents latitude. By using one, we can’t find relative position of a word. The number 10000 represents wavelength. Scientist use this value because circle completes after very long run, means value does not repeat after short time. In some model it is 100000, 500000, so it depends on usage of model. D_model represents dimensions of model. 

If we have eight values, the model will run for i=0-3 as in one iteration it fills two slots, sin for odd and cos for even slot. 
For example. Pos = 2, i=0 (slot 0 and 1), d_model = 8
For i=0
Q = pos/10000^2i/d_model
Q = 2/ 10000^0/8
Q= 2
Now:
Slot 0: PE(2,0) = sin(2) 
Slot 1: PE(2,1) = cos(2)
Then it will continue for i=1,2,3. Let's see for last iteration
For i=3
Q3 = 2/10000^6/8
Q3 = 2/1000
Q3 = 0.002
Slot 6: PE(2,6) = sin(0.002)
Slot 7: PE(2,7) = cos(0.002)

PE for pos 2 = [sin(2), cos(2),..........,sin(0.002),cos(0.002)]

As these vector’s dimensions are same compare to word embedding matrix, so we perform element wise addition.

**Final Embedding = Word Embedding + Positional Embedding**

