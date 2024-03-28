|序号|公式|规则|
|---|---|---|
|1|a &rightarrow; ((b &rightarrow; a) &rightarrow; a)|L1|
|2|(a &rightarrow; ((b &rightarrow; a) &rightarrow; a)) &rightarrow; ((a &rightarrow; (b &rightarrow; a)) &rightarrow; (a &rightarrow; a))|L2|
|3|(a &rightarrow; (b &rightarrow; a)) &rightarrow; (a &rightarrow; a)|(1), (2), MP|
|4|a &rightarrow; (b &rightarrow; a)|L1|
|5|a &rightarrow; a|(3), (4), MP|
|6|a &rightarrow; (a &rightarrow; b)|假定|
|7|(a &rightarrow; (a &rightarrow; b)) &rightarrow; ((a &rightarrow; a) &rightarrow; (a &rightarrow; b))|L2|
|8|(a &rightarrow; a) &rightarrow; (a &rightarrow; b)|(6), (7), MP|
|9|a &rightarrow; b|(5), (8), MP|
