# Spam filtering with Naïve Bayes

## The Problem

Suppose we are receving lots of E-mails every day and a number of them are spam emails, we want to classify whether the E-mail is a spam or not. To help solve the problem, we have already prepared some E-mails with labels telling us whether one is spam or not.

## The Algorithm

Firstly, we construct a vocabulary $V = \{v_1, v_2, v_3, \ldots,v_n\}$ from the training dataset by collecting all the unique words in all emails and assigning them an index $i \in \mathbf{N}$, so that for a word $w$ labeled $i$, $v_i=w$, where $n$ is the total number of unique words.

For an E-mail $D$ with words $W=\{w_1, w_2, \ldots, w_{n_d}\}$, where $n_d$ is the number of words in the email, we can extract its feature by counting frequencies of each word and form a vector $\textbf{x} = \langle x_1, x_2, x_3,\ldots,x_{|V|}\rangle \in \mathbf{Z}^{|V|}$, where $x_i = count_{w \in W}(w=v_i)$. We assign a class $y \in \mathbf{C}$ to each email, where $\mathbf{C}=\{c_1, c_2, \ldots, c_{n_c}\}$ and $n_c$ is the number of unique labels. Then we can construct sample set  $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2\ldots,\mathbf{x}_{n_x}\}$ where $n_x$ is number of training samples and ground truths set $\mathbf{Y}=\{y_1, y_2, \ldots, y_{n_x}\}$.

For an E-mail $D$ with words $W=\{w_1, w_2, \ldots, w_{n_d}\}$, the probability of belonging to class $c_i$ is:

$$
P(c_i|w_1, w_2, \ldots, w_{n_d})=\frac{P(w_1, w_2, \ldots, w_{n_d}|c_i)P(c_i)}{P(w_1, w_2, \ldots, w_{n_d})}
$$

To determine the class the E-mail belongs to, we only need to take argmax of every probability:

$$
\hat{y}=\argmax_{c}P(c|w_1, w_2, \ldots, w_{n_d})
$$

Because $P(w_1, w_2, \ldots, w_{n_d})$ is constant, we can omit the item from denumerator. And according to the naïve assumption that words are independent, we can simply the formula to:

$$
\hat{y}=\argmax_{c}P(c)\prod^{n_d}_{i=1}P(w_i|c)
$$

It is easy to compute $P(c_i)$:

$$
P(c_i)=\frac{count_{y \in \mathbf{Y}}(y=c_i)}{|\mathbf{Y}|}
$$

To compute $P(w|c)$:

$$
P(w_i|c)=\frac{count_{all\ words}(w=w_i, y=c)}{count_{all\ words}(y=c)}
$$

However, for a word $w_i$ that never appears in class $c$, the probability will become 0 which is unreasonable. So we can assume a new word `<UNK>`, and map all those words to `<UNK>`. We can achieve this by adding a smoothing factor $0\lt\alpha\le1$ to the probability:

$$
P(w_i|c)=\frac{count_{all\ words}(w=w_i, y=c)+\alpha}{count_{all\ words}(y=c)+\alpha(|V|+1)}
$$

Moreover, to avoid floating point precision problem, we take the logarithm of the probability and add them up instead of multiplying them:

$$
\hat{y}=\argmax_{c}\log P(c)\sum^{n_d}_{i=1}\log P(w_i|c)
$$

This is our Naïve Bayes classifier model.

## How to Run

```bash
python3 main.py --data="-400" --alpha=0.01
```

Arguments:

- --data: Specify dataset used for training, choose from `""`, `"-400"`, `"-100"`, `"-50"`, default: `""`
- --alpha: Specify smoothing factor of the model, should be greater than 0 and not greater than 1, default: `0.001`
