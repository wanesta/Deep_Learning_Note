**Math**

$\sum_{i=0}^{100} i$ $f(x) = sin(x) + 12 $ $Wt=\prod \sum_{sf}^{s}$ $$ \begin{align} 2x - 5y = 8 3x + 9y = -12 \end{align} $$ ![cat](assets/cat.jpg) <https://shd101wyy.github.io/markdown-preview-enhanced/#/zh-cn/markdown-basics> ![GitHub Logo](assets/cat.jpg) ![Alt Text](http://img4.imgtn.bdimg.com/it/u=3432487329,2901563519&fm=26&gp=0.jpg) ![cat](assets/cat.jpg)

> We're living the future so the present is our past.

如下，三个或者更多的

--------------------------------------------------------------------------------

连字符

--------------------------------------------------------------------------------

星号

--------------------------------------------------------------------------------

下划线

Content [^1]

[^1]\: Hi! This is a footnote

`<addr>` 才对。

```{gnuplot output:"html", id:"chj3p9gbg3"} set terminal svg set title "Simple Plots" font ",20" set key left box set samples 50 set style data points

plot [-10:10] sin(x),atan(x),cos(atan(x))

````

```python3 {.lineNo}
print("test !")
````

```{python matplotlib:true, id:"chj3p9eitf"} import matplotlib.pyplot as plt plt.plot([1,2,3, 4]) plt.show()

````

| First Header                  | Second Header                |
| ----------------------------- | ---------------------------- |
| Content from cell 1         区 | Content from cell 2          |
| Content in the first column   | Content in the second column |

```{gnuplot output:"html", id:"chj3p946sy"}
set terminal svg
set title "simple Plots" font ",20"
set key loft box
set samples 50
set style data points
plot [-10:10] sin(x), atan(x),cos(atan(x))
````

@import "<https://cdn.plot.ly/plotly-latest.min.js>"

````
```

```mermaid
graph TD;
    A-->B
    A-->C
    B-->D
    C-->D
````

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

```mermaid
%% Example of sequence diagram
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
```

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

```mermaid
graph TD
    A-->|read / write|center((center server))
    B-->|read only|center
    C-->|read / write|center
    D-->|read only|center
    E(...)-->center
    center---|store the data|sharedrive
```

```mermaid
%% Example of sequence diagram
sequenceDiagram
    Alice->John: Hello John, how are you?
    loop Reply every minute
        John-->Alice: Great!
    end
```

===============================@@@@@@@@@@@@

```mermaid
sequenceDiagram
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
```

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

```mermaid
graph LR
    A[Square Rect] -- Link text --> B((Circle))
    A --> C(Round Rect)
    B --> D{Rhombus}
    C --> D
```

```mermaid
        gantt
        dateFormat  YYYY-MM-DD
        title Adding GANTT diagram functionality to mermaid

        section A section
        Completed task            :done,    des1, 2014-01-06,2014-01-08
        Active task               :active,  des2, 2014-01-09, 3d
        Future task               :         des3, after des2, 5d
        Future task2              :         des4, after des3, 5d

        section Critical tasks
        Completed task in the critical line :crit, done, 2014-01-06,24h
        Implement parser and jison          :crit, done, after des1, 2d
        Create tests for parser             :crit, active, 3d
        Future task in critical line        :crit, 5d
        Create tests for renderer           :2d
        Add to mermaid                      :1d

        section Documentation
        Describe gantt syntax               :active, a1, after des1, 3d
        Add gantt diagram to demo page      :after a1  , 20h
        Add another diagram to demo page    :doc1, after a1  , 48h

        section Last section
        Describe gantt syntax               :after doc1, 3d
        Add gantt diagram to demo page      :20h
        Add another diagram to demo page    :48h
```

```mermaid
graph TB
    sq[Square shape] --> ci((Circle shape))

    subgraph A subgraph
        od>Odd shape]-- Two line<br>edge comment --> ro
        di{Diamond with <br/> line break} -.-> ro(Rounded<br>square<br>shape)
        di==>ro2(Rounded square shape)
    end

    %% Notice that no text in shape are added here instead that is appended further down
    e --> od3>Really long text with linebreak<br>in an Odd shape]

    %% Comments after double percent signs
    e((Inner / circle<br>and some odd <br>special characters)) --> f(,.?!+-*ز)

    cyr[Cyrillic]-->cyr2((Circle shape Начало));

     classDef green fill:#9f6,stroke:#333,stroke-width:2px;
     classDef orange fill:#f96,stroke:#333,stroke-width:4px;
     class sq,e green
     class di orange
```
