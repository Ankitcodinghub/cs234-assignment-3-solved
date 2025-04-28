# cs234-assignment-3-solved
**TO GET THIS SOLUTION VISIT:** [CS234 Assignment #3 Solved](https://www.ankitcodinghub.com/product/cs-234-assignment-3-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118626&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS234 Assignment #3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
These questions require thought, but do not require long answers. Please be as concise as possible.

Please review any additional instructions posted on the assignment page at http://web.stanford.edu/class/cs234/assignments.html. When you are ready to submit, please follow the instructions on the course website.

1 Policy Gradient Methods (50 pts coding + 15 pts writeup)

The goal of this problem is to experiment with policy gradient and its variants, including variance reduction methods. Your goals will be to set up policy gradient for both continuous and discrete environments, and implement a neural network baseline for variance reduction. The framework for the policy gradient algorithm is setup in main.py, and everything that you need to implement is in the files networkutils.py, policy.py, policy gradient.py and baselinenetwork.py. The file has detailed instructions for each implementation task, but an overview of key steps in the algorithm is provided here.

1.1 REINFORCE

Recall the policy gradient theorem,

∇θJ(θ) = Eπθ [∇θ logπθ(a|s)Qπθ(s,a)]

REINFORCE is a Monte Carlo policy gradient algorithm, so we will be using the sampled returns Gt as unbiased estimates of Qπθ(s,a). The REINFORCE estimator can be expressed as the gradient of the following objective function:

where D is the set of all trajectories collected by policy πθ, and ) is trajectory i.

1.2 Baseline

One difficulty of training with the REINFORCE algorithm is that the Monte Carlo sampled return(s) Gt can have high variance. To reduce variance, we subtract a baseline bφ(s) from the estimated returns when computing the policy gradient. A good baseline is the state value function, V πθ(s), which requires a training

1

update to φ to minimize the following mean-squared error loss:

LMSE

1.3 Advantage Normalization

After subtracting the baseline, we get the following new objective function:

where

A second variance reduction technique is to normalize the computed advantages, Aˆit, so that they have mean 0 and standard deviation 1. From a theoretical perspective, we can consider centering the advantages to be simply adjusting the advantages by a constant baseline, which does not change the policy gradient. Likewise, rescaling the advantages effectively changes the learning rate by a factor of 1/σ, where σ is the standard deviation of the empirical advantages.

1.4 Coding Questions (50 pts)

The functions that you need to implement in networkutils.py, policy.py, policygradient.py, and baselinenetwork.py are enumerated here. Detailed instructions for each function can be found in the comments in each of these files.

Note: The ”batch size” for all the arguments is PTi since we already flattened out all the episode observations, actions, and rewards for you. In networkutils.py,

• buildmlp

In policy.py,

• BasePolicy.act

• CategoricalPolicy.actiondistribution

• GaussianPolicy. init

• GaussianPolicy.std

• GaussianPolicy.actiondistribution

In policygradient.py,

• PolicyGradient.initpolicy

• PolicyGradient.getreturns

• PolicyGradient.normalizeadvantage

• PolicyGradient.updatepolicy

In baselinenetwork.py,

• BaselineNetwork. init

• BaselineNetwork.forward

• BaselineNetwork.calculateadvantage

• BaselineNetwork.updatebaseline

1.5 Testing

You can also add additional tests of your own design in tests/testbasic.py.

1.6 Writeup Questions (15 pts)

(a) (3 pts) To compute the REINFORCE estimator, you will need to calculate the values (we drop the trajectory index i for simplicity), where

Naively, computing all these values takes O(T2) time. Describe how to compute them in O(T) time.

(b) (12 pts) The general form for running your policy gradient implementation is as follows:

if not using a baseline, or

if using a baseline. Here ENV should be cartpole, pendulum, or cheetah, and SEED should be a positive integer.

For each of the 3 environments, choose 3 random seeds and run the algorithm both without baseline and with baseline. Then plot the results using

where SEEDS should be a comma-separated list of seeds which you want to plot (e.g. –seeds 1,2,3). Please include the plots (one for each environment) in your writeup, and comment on whether or not you observe improved performance when using a baseline.

We have the following expectations about performance to receive full credit:

• cheetah: Should reach at least 200 (Could be as large as 950)

2 Reducing Variance in Policy Gradient Methods (35 pts)

In class, we explored REINFORCE as a policy gradient method with no bias but high variance. In this problem, we will explore methods to dramatically reduce variance in policy gradient methods, potentially at the cost of increased bias.

Let us consider an infinite horizon MDP M = hS,A,R,T ,γi. Let us define

Aπ(st,at) = Qπ(st,at) − V π(st)

An approximation to the policy gradient is defined as (1)

∞

g = Es0:∞[XAπ(st,at)∇θ log πθ(at,st)]

a0:∞ t=0

where the colon notation a : b represents the range [a,a + 1,a + 2,…b] inclusive of both ends. (2)

(a) (5 pts) Let us define the partial sum . Show that it is not necessarily true that Var(Rt+1) ≥ Var(Rt). [Hint: Construct a counterexample MDP where this statement does not hold.]

(b) (10 pts) Prove that Var(Rt+1) ≥ Var(Rt) is true if we assume that rt+1 is, on average, correlated with the previous rewards, i.e. Cov(ri,rt+1) &gt; 0.

(c) (5 pts) In practice, we do not have access to the true function Aπ(st,at), so we would like to obtain an estimate instead. We will consider the general form of an estimator Aˆt(s0:∞,a0:∞) that can be a function of the entire trajectory.

Let Aˆt(s0:∞,a0:∞) = Qˆt(st:∞,at:∞)−bt(s0:t,a0:t−1), where for all st,at, we have that Qˆt is an unbiased estimator of the true Qπ. Namely, we have that ). Note that bt is an arbitrary function of the actions and states sampled before at. Prove that by using this estimate of Aˆt, we obtain an unbiased estimate of the policy gradient g. In other words, prove that

.

(d) (5 pts) We will now look at a few different variants of Aˆt. Recall the TD error δtVˆ (st,at) = rt+γVˆ(st+1)−Vˆ(st). If Vˆ = V π, prove that is an unbiased estimate of Aπ.

(e) (5 pts) Let us define . Show that . In general,

how does bias and variance change as k increases? (a few sentences of justification would suffice, no formal proof is necessary)

(f) (5 pts) Show that
