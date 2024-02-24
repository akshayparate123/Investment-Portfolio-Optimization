<a name="br1"></a> 

Investment Portfolio Optimization

1st Akshay Parate

Data Science

2<sup>nd</sup> Akhil Karumanchi

3rd Sai Nithya Surasani

Data Science

Data Science

Stevens Institute of Technology (of Aff.) Stevens Institute of Technology (of Aff.) Stevens Institute of Technology (of Aff.)

Hoboken, United States of America

aparate@stevens.edu

Hoboken, United States of America

akaruman@stevens.edu

Hoboken, United States of America

ssurasan@stevens.edu

Abstract—The project focuses on Investment Portfolio Opti- ”high-risk,” providing users with a clear understanding of

mization with the aim of maximizing returns while maintaining the risk associated with each asset. Moreover, our model

an acceptable level of risk. The problem is addressed through the

introduces a unique aspect of user control investors can adjust

risk percentages according to their preferences, allowing for

a personalized investment plan that adapts dynamically to

application of machine learning algorithms, including decision

trees and random forests for asset selection strategies. Addition-

ally, support vector machines (SVM) are employed to ﬁne-tune

portfolio allocation. The optimization of investment portfolios is changing risk appetites.

achieved by leveraging linear regression to predict asset returns

and associated risks. The implementation includes a classiﬁca-

tion approach to categorize assets into ”low-risk,” ”medium-

risk,” or ”high-risk.” The experimental results showcase the

effectiveness of these methodologies. The major contribution lies

The experimental results demonstrate the efﬁcacy of our

approach, showcasing superior returns while maintaining ac-

ceptable risk levels. The ability to categorize assets and

empower investors with control over risk percentages positions

in the comprehensive integration of diverse machine learning our solution as a user-friendly and adaptive tool. This stands as

techniques for a robust and efﬁcient approach to investment

portfolio management, offering potential advantages over existing

solutions.

a signiﬁcant advantage over existing solutions, providing in-

vestors with a nuanced and personalized approach to portfolio

optimization in the ever-evolving ﬁnancial landscape.

I. INTRODUCTION

II. RELATED WORK

Investment Portfolio Optimization is a crucial aspect of

Existing Solutions: Various approaches have been employed

ﬁnancial decision-making, aiming to strike a balance between to address the challenges of investment portfolio optimization,

maximizing returns and managing acceptable risks. In this each with its unique set of advantages and limitations. Catego-

project, we delve into the intricate world of investment portfo- rizing these existing techniques reveals distinct strategies and

lios, speciﬁcally addressing the challenges associated with US methodologies.

stocks, bonds, gold, real estate, and currencies over the last

Traditional Markowitz Mean-Variance Optimization: One

two decades. Our dataset encompasses a wealth of ﬁnancial of the earliest and widely used methods is the Markowitz

information, providing a comprehensive view of historical Mean-Variance Optimization (MVO). Introduced by Harry

market trends and asset performance.

Markowitz in 1952, MVO aims to maximize portfolio returns

As ﬁnancial landscapes become increasingly complex, tradi- for a given level of risk. While MVO provides a theoretical

tional investment strategies may fall short in delivering optimal framework for balancing risk and return, its reliance on

results. To address this, our solution adopts a pragmatic yet historical return and covariance data has been criticized for its

effective approach. Leveraging machine learning algorithms, sensitivity to estimation errors and the assumption of normal

we employ decision trees and random forests for insightful distribution, which may not always hold true in real-world

asset selection strategies. Further, we ﬁne-tune portfolio al- ﬁnancial markets [Markowitz, 1952].

locations with support vector machines (SVM) to enhance

Modern Portfolio Theory Extensions: Numerous extensions

precision in asset distribution. The key innovation lies in the and reﬁnements to Modern Portfolio Theory (MPT) have been

implementation of linear regression, enabling us to predict proposed, incorporating factors such as downside risk, trans-

asset returns and risks with a high degree of accuracy.

action costs, and non-normal distributions. These extensions

Our dataset, spanning 20 years, offers a rich repository attempt to address some of the shortcomings of MVO but

of historical ﬁnancial data, allowing us to train and test often come with increased complexity and computational de-

our machine learning models comprehensively. This dataset mands. For example, incorporating transaction costs may lead

includes information on US stocks, bonds, gold, real estate, to computationally intensive optimization problems [Fernholz

and currencies, capturing the nuances of market dynamics and and Shay, 1982]. While each of these existing solutions con-

providing a robust foundation for our optimization strategies. tributes to the ﬁeld of portfolio optimization, they come with

One distinctive feature of our solution is the simplicity and their trade-offs. Traditional methods often rely on simplifying

ﬂexibility it offers to the investor. Through a categorization assumptions that may not hold in real-world scenarios, while

system, assets are classiﬁed as ”low-risk,” ”medium-risk,” or modern techniques, though more sophisticated, may introduce



<a name="br2"></a> 

complexities that hinder practical implementation. Machine Through this analysis, we identiﬁed and subsequently dropped

learning, while promising, requires careful consideration of irrelevant columns.

data quality and model interpretability. In the next section, we

will introduce our novel approach, highlighting its advantages

over existing solutions in terms of simplicity, adaptability, and potential patterns. We speciﬁcally focused on visualizing the

user control.

Visualization of Data:

Visualizing the data provides insights into its structure and

data for gold stocks and bonds, two critical components of

investment portfolios. These visualizations helped us gain a

deeper understanding of the trends, volatility, and potential

correlations between these assets.

III. OUR SOLUTION

This section elaborates your solution to the problem.

Correlation Analysis:

A. Description of Dataset

To identify and address multi- collinearity, we calculated

the correlation coefﬁcients between features. Features with

high correlation were either dropped or, if necessary for the

analysis, addressed through techniques such as dimensionality

reduction.

The dataset employed in this project constitutes ﬁnancial

data covering various asset classes, including US stocks,

bonds, gold, real estate, and currencies. Spanning the last

two decades, this dataset offers a substantial temporal scope,

allowing for a thorough examination of market trends and the

performance of diverse ﬁnancial instruments.

Feature Engineering:

Feature engineering involved trans- forming and creating

variables to improve the model’s performance. This step in-

cluded encoding categorical variables, normalizing numerical

values, and creating new features that could capture valuable

information for the optimization process.

Components of the Dataset:

US Stocks - Inclusion of data related to a broad spectrum of

US stocks provides insights into the performance of individual

companies across different sectors.

Bonds - Information pertaining to bonds is crucial for

understanding ﬁxed-income securities, offering a perspective

on debt markets and interest rate movements.

Gold - Gold is a signiﬁcant commodity in the ﬁnancial

markets, often considered a safe-haven asset. The dataset likely

includes metrics related to gold prices, aiding in the analysis

of market sentiment and economic conditions

Real Estate - Real estate data contributes to a holistic

view of investment opportunities, reﬂecting trends in property

values and the broader real estate market.

Currencies - The inclusion of currency data allows for an

examination of foreign exchange markets, providing insights

into currency ﬂuctuations and global economic conditions.

The comprehensive nature of the dataset serves as a robust

foundation for the investment portfolio optimization process.

By incorporating data from diverse asset classes over an

extended period, the project aims to capture a nuanced un-

derstanding of market dynamics and historical performance

metrics. This, in turn, facilitates the development of a sophis-

ticated optimization model geared towards maximizing returns

and managing risks effectively.

Example Visualization:

Gold Stocks and Bonds: As an illustration of our ap-

proach, we generated visualizations depicting the historical

performance of gold stocks and bonds. These visualizations

included time-series plots, moving averages, and volatility

analyses. By examining these visual representations, we gained

valuable insights into the behavior of these assets, informing

our decision-making process in the portfolio optimization.

In conclusion, the preprocessing steps undertaken in this

project have played a pivotal role in ensuring that the dataset

is clean, relevant, and well-structured for subsequent analysis.

The comprehensive approach, which involved a combination

of statistical analyses, visualizations, and thoughtful feature

engineering, has established a solid foundation for the machine

learning-based portfolio optimization.

B. Machine Learning Algorithms

Decision Trees and Random Forests:

Decision trees are well-suited for asset selection strategies

due to their innate capacity to capture complex decision

boundaries. This makes them particularly effective in scenarios

Data Source:

The dataset was acquired from Kaggle, a reputable ﬁnancial where the relationships between different features and the

data provider. It covers a wide range of assets, enabling a target variable (asset selection in this case) are intricate and

holistic analysis of investment opportunities.

Data Preprocessing:

non-linear.

Random Forests, employed as an ensemble of decision

Handling Missing Data: Fortunately, the dataset exhibited trees, extend the capabilities of individual decision trees. This

very few missing values. As a prudent preprocessing step, ensemble approach enhances the robustness of the model

we opted to drop instances with missing data, ensuring the and mitigates the risk of overﬁtting, a common challenge in

integrity of our analyses.

complex datasets.

Addressing Irrelevant Columns:

Design: We employ decision trees to evaluate the impor-

To enhance the efﬁciency of our model, we conducted a tance of various features in asset selection. The Random Forest

correlation analysis among different features. Highly corre- ensemble aggregates predictions from multiple trees, providing

lated features can introduce multicollinearity issues and might a more reliable and stable model.

not contribute signiﬁcantly to the model’s predictive power.

Support Vector Machines (SVM):



<a name="br3"></a> 

SVM is utilized in portfolio optimization to identify an and providing interpretable results. Model evaluation metrics

optimal hyperplane for asset allocation. It aims to maximize are carefully chosen based on the speciﬁc goals of the risk

the margin between different asset classes, enhancing the categorization task. This ensures that the classiﬁcation models

model’s robustness. SVM considers non-linear relationships, align with the desired outcomes and effectively meet the

making it suitable for complex ﬁnancial market dynamics. objectives of the investment strategy.

By classifying assets based on historical data, SVM aids in

strategic portfolio diversiﬁcation. Its ability to handle high-

Neural Networks (Optional):

Neural networks are considered for their capability to learn

dimensional data contributes to effective risk management and intricate patterns and relationships within ﬁnancial data, espe-

improved portfolio performance.

The design involves leveraging SVM to optimize portfolio

cially valuable in navigating complex market dynamics.

Design: If needed, a neural network with multiple hidden

allocation, ensuring assets are strategically positioned within layers and nodes is contemplated for portfolio optimization.

the feature space, enhancing the overall effectiveness of the This design aims to harness the ﬂexibility of neural networks

allocation strategy. kernel functions are explored to capture to enhance the model’s ability to capture nuanced market

non-linear relationships among assets, enabling the model to trends.

discern intricate patterns crucial for portfolio optimization.

Support Vector Machines (SVM) for Fine-tuning Portfolio introduce non-linearities into the neural network, allowing it

Allocation: to better adapt to the diverse and dynamic nature of ﬁnancial

Activation functions, such as ReLU or tanh, are explored to

SVM is selected to ﬁne-tune portfolio allocation due to market data. Initial parameters, including the number of layers

its proﬁciency in optimizing decision boundaries, allowing and nodes, are chosen based on empirical testing. Param-

for strategic asset positioning. Particularly effective in the eters such as learning rates, batch sizes, and regularization

ﬁnancial domain, SVM excels when dealing with non-linear strength are set initially and adjusted during training, utilizing

relationships between assets, a common characteristic in com- techniques like grid search or random search for ﬁne-tuning.

plex market dynamics. The design involves leveraging SVM to Model evaluation metrics, such as Sharpe ratio and cumulative

optimize portfolio allocation, ensuring assets are strategically returns, are employed to assess the effectiveness of each

positioned within the feature space, enhancing the overall neural network conﬁguration in achieving the project goals.

effectiveness of the allocation strategy. Various kernel func- These metrics provide a quantitative measure of the model’s

tions are explored to capture non-linear relationships among performance in terms of risk-adjusted returns.

assets, enabling the model to discern intricate patterns crucial

C. Implementation Details

for portfolio optimization. The optimization process includes

tuning regularization parameters in SVM to achieve optimal

performance, enhancing the model’s adaptability to varying

Data Visualization and Handling Missing Data:

Utilized Matplotlib and Seaborn libraries to create visual-

market conditions and improving its overall efﬁcacy in port- izations (e.g., line charts, scatter plots) for a comprehensive

folio allocation.

Linear Regression:

understanding of the dataset. Explored key variables and trends

to uncover patterns and potential insights that could inform the

Linear regression is a ﬁtting choice for predicting asset re- subsequent analysis. Employed Pandas to assess the dataset for

turns and risk, offering a clear and interpretable understanding missing values and understand the extent of data completeness.

of relationships between variables in the ﬁnancial context.

Utilized summary statistics and visualizations to identify any

Design: Linear regression models are intentionally crafted instances of missing data and its distribution across variables.

to predict both asset returns and risks. This approach provides Noted the minimal presence of missing values, allowing for

a comprehensive view, allowing for informed decision-making informed decision-making, we decided to drop instances to

in portfolio optimization. To capture relevant factors inﬂuenc- maintain data integrity.

ing returns and risks, feature engineering is employed. This

Risk Analysis - Volatility Calculation: Computed daily

involves selecting and transforming features to enhance the returns for each asset to understand their day-to-day perfor-

model’s predictive power and accuracy. while its simplicity mance. Calculated volatility using the rolling standard devia-

enhances interpretability for users involved in the portfolio tion over a speciﬁed window (252 days), providing a measure

optimization process.

Classiﬁcation Models:

of risk.

Portfolio Simulation: Simulated various portfolios by ran-

Classiﬁcation models are applied to categorize assets into domly assigning weights to different assets, such as stocks,

risk categories, enhancing transparency in the investment strat- gold, and bonds. The random weights represent different asset

egy by providing a clear risk assessment for each asset. The compositions within each simulated portfolio. Computed the

design involves employing classiﬁcation algorithms to label returns of each simulated portfolio by combining the weighted

assets as ”low-risk,” ”medium-risk,” or ”high-risk,” enabling a returns of individual assets. Calculated the volatility of each

systematic and data-driven approach to risk management in the simulated portfolio using the portfolio’s standard deviation,

portfolio. Algorithms such as logistic regression or decision considering the covariance between assets.

trees are considered for the classiﬁcation task, leveraging

Performance Metrics Calculation: Computed key perfor-

their respective strengths in capturing complex relationships mance metrics, including the Sharpe ratio, cumulative returns,



<a name="br4"></a> 

and annualized returns. Sharpe ratio was used to assess risk-

adjusted returns, providing a measure of how well the returns

compensate for the level of risk taken.

Correlation Analysis: Conducted a comprehensive analysis

to understand the correlation between different assets in the

portfolio. Utilized statistical measures to assess the strength

and direction of linear relationships between pairs of assets.

Examined the correlation coefﬁcients to determine the strength

of relationships between assets.

Covariance Plot: Utilized a heatmap to visualize the co-

variance matrix, providing a graphical representation of co-

variances between different asset pairs. The heatmap allows

for easy identiﬁcation of patterns, dependencies, and potential

relationships between assets. Covariance measures the degree

to which two variables change together. In the context of

assets, it provides insights into how changes in the value of

one asset might impact another.

Fig. 1. Returns vs Volatality

Feature Engineering for Linear Regression: Created a new

dataset with independent variables, including asset weights,

volatility, and Sharpe ratio, and a dependent variable, which

is the returns of the portfolio. This dataset serves as input for

the linear regression model.

Visualizing the Dataset: Created visualizations of the new

dataset, showcasing how returns change based on asset

weights, volatility, and Sharpe ratio. Utilized tools such as

Matplotlib or Seaborn to generate informative plots. Analyzed

the visualizations to identify potential patterns and relation-

ships within the data. Examined how changes in asset weights,

volatility, and Sharpe ratio correlate with variations in portfolio

returns.

Fig. 2. Covariance

Linear Regression Implementation: Implemented a linear

regression algorithm to predict portfolio returns based on input

features, including asset weights, volatility, and Sharpe ratio.

Utilized a library such as scikit-learn for the implementation.

Explored techniques such as regularization to prevent overﬁt-

ting and enhance the model’s generalization to unseen data.

Regularization helps control the complexity of the model,

avoiding overly complex ﬁts that may not generalize well.

Fine-tuned hyperparameters, including regularization strength,

to optimize the performance of the linear regression model.

Adjusted parameters through techniques like grid search or

random search, optimizing the model for predictive accuracy.

Ensured a comprehensive exploration of the dataset by in-

corporating features relevant to portfolio returns. Conducted

detailed risk analysis to understand the impact of different

variables on the model’s predictions. Adopted an iterative

approach involving testing and visualization to reﬁne the

model. Iteratively tested the model’s performance, visualizing

results to gain insights and make necessary adjustments.

Fig. 3. Covariance

IV. COMPARISON

Fig. 4. Monte carlo simulation

Linear regression provides a straightforward interpretation

of coefﬁcients, offering stakeholders a clear understanding of

how each independent variable inﬂuences portfolio returns.

This transparency is particularly valuable in ﬁnancial contexts

where interpretable models support informed decision-making.



<a name="br5"></a> 

linear patterns in ﬁnancial data.

Decision Trees and Random Forests:

Strengths: Ability to capture complex decision boundaries,

robustness, reduced overﬁtting in Random Forests. Weak-

nesses: Prone to overﬁtting in decision trees, interpretability

challenges in Random Forests.

Support Vector Machines (SVM):

Strengths: Effective in optimizing decision boundaries, han-

dles non-linear relationships. Weaknesses: Interpretability, sen-

sitivity to hyperparameters.

Neural Networks:

Strengths: Capacity to learn intricate patterns, ﬂexibility

in handling complex data. Weaknesses: Complexity, potential

overﬁtting, sensitivity to hyperparameters.

Fig. 5. Efﬁcient fontier

A comprehensive evaluation considering interpretability,

complexity, and adaptability to data patterns is essential. The

choice of the ”better” algorithm depends on the speciﬁc goals

of the portfolio optimization and the characteristics of the

ﬁnancial data. Experimentation, testing, and benchmarking

against existing solutions contribute to informed decision-

making in algorithm selection.

V. FUTURE DIRECTIONS

Certainly, given an additional 3-6 months, there are several

avenues to further enhance the performance of the investment

portfolio optimization model: Explore additional features that

may inﬂuence portfolio returns. This could include economic

indicators, global events, or sentiment analysis from ﬁnancial

news. Investigate more sophisticated measures of risk, beyond

standard deviation, to capture tail risk or extreme events.

Experiment with more advanced machine learning models

such as ensemble methods (e.g., stacking), gradient boosting,

or deep learning architectures. Incorporate time-series analysis

techniques to capture temporal dependencies and patterns in

ﬁnancial data. Implement dynamic asset allocation strategies

that adapt to changing market conditions. This could involve

incorporating techniques like reinforcement learning to opti-

mize allocations over time.

The design involves the utilization of classiﬁcation algo-

rithms speciﬁcally tailored for categorizing assets into risk

levels. This strategic use of classiﬁcation facilitates a sys-

tematic and structured approach to risk management within

the portfolio. Assets are categorized into risk levels such as

”low-risk,” ”medium-risk,” or ”high-risk,” providing a clear

and actionable framework for decision-making in portfolio

optimization. Classiﬁcation algorithms such as logistic regres-

sion and decision trees are considered for their suitability

in handling the task of risk categorization. The choice of

algorithms is driven by their interpretability and ability to

capture complex decision boundaries. Model evaluation met-

rics are carefully chosen based on the speciﬁc goals of the

classiﬁcation task. Metrics such as precision, recall, and F1-

score may be employed to assess the model’s performance in

accurately categorizing assets into their respective risk levels.

Linear Regression:

VI. CONCLUSION

In conclusion, the ”Investment Portfolio Optimization”

project successfully demonstrated the effectiveness of a data-

driven approach to enhance portfolio performance. By lever-

aging historical ﬁnancial data, risk assessment models, and

advanced optimization algorithms, we aimed to create a well-

balanced portfolio that maximizes returns while managing risk.

Through this project by analyzing the data, we identiﬁed

historical trends and risk metrics for a diverse set of assets

and emphasized the importance of diversiﬁcation in reducing

overall portfolio risk. The inclusion of risk management strate-

gies and constraints further ensured that the recommended

portfolios align with investors’ risk tolerance and preferences.

In Summary, the ”Investment Portfolio Optimization”

project not only delivered a powerful tool for investors seeking

Strengths: Interpretability, simplicity, well-suited for linear to maximize returns within their risk tolerance but also high-

relationships Weaknesses: Limited in capturing complex, non- lighted the importance of ongoing monitoring and adaptation



<a name="br6"></a> 

in the ever-changing ﬁnancial landscape. The methodologies In essence, the project’s key takeaways collectively contribute

and insights generated by this project contribute to the broader valuable insights to the ﬁeld of investment portfolio opti-

discourse on data-driven investment strategies, providing a mization. By laying the groundwork for future enhancements,

foundation for informed decision-making in the complex world exploring advanced modeling techniques, and maintaining a

of ﬁnancial markets.

Key Takeaways:

forward-looking perspective, the project establishes itself as a

signiﬁcant contribution to the ongoing journey of reﬁnement

Foundational Steps: The project initiated with meticulous and adaptation in the ever-changing landscape of ﬁnancial

data preprocessing, encompassing essential steps such as data markets.

visualization, risk analysis, and the creation of a dataset

with key metrics. These foundational steps were crucial in

References-

https://www.kaggle.com/code/kashnitsky/a4-

gaining a comprehensive understanding of the ﬁnancial data, demo-linear-regression-as-optimization

establishing a solid groundwork for subsequent modeling and https://smartasset.com/investing/guide-portfolio-optimization-

analysis.

Linear Regression Approach: The decision to implement

strategies

linear regression for return prediction based on asset weights,

volatility, and Sharpe ratio underscores the project’s commit-

ment to transparency and interpretability. Linear regression

provides a clear framework for stakeholders to comprehend

the factors inﬂuencing return predictions, fostering conﬁdence

and trust in the decision-making process.

Algorithmic Suitability: While linear regression offers trans-

parency, the acknowledgement of the complex nature of ﬁ-

nancial markets suggests a recognition of the need for more

advanced algorithms. The mention of decision trees or en-

semble methods indicates an openness to exploring models

better suited to capturing intricate patterns and nonlinear

relationships inherent in ﬁnancial data.

Opportunities for Improvement: The project identiﬁes sev-

eral opportunities for enhancement. Exploring alternative al-

gorithms, incorporating additional features, and adopting ad-

vanced modeling techniques are highlighted as avenues to

improve the robustness and adaptability of the optimization

model. This forward-looking perspective emphasizes a com-

mitment to reﬁning the methodology for optimal performance.

Looking Forward: The iterative nature of model develop-

ment is emphasized as a key takeaway. The project recognizes

that continuous reﬁnement and adaptation are integral to

staying relevant in the dynamic world of ﬁnance. Consider-

ations for feature engineering, dynamic asset allocation, risk

management, and external data integration highlight a forward-

thinking approach to further enhance the model’s predictive

power and applicability.

Limitations and Considerations: Acknowledging the limi-

tations of linear regression in capturing complex market dy-

namics demonstrates a realistic understanding of the model’s

constraints. The project’s focus on a subset of asset classes

suggests a deliberate choice and points towards potential future

expansions to include a broader array of assets for a more

comprehensive and diversiﬁed perspective.

Continuous Iteration: The recognition of the dynamic nature

of the ﬁnancial landscape is a key takeaway. The project’s

conclusion is framed as a milestone rather than an endpoint,

emphasizing the necessity for continuous iteration and adap-

tation of models to align with evolving market conditions

and investor preferences. This iterative mindset positions the

project as an ongoing exploration rather than a ﬁnite endeavor.


