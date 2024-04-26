# Module 18 Challenge - PyChain Ledger
<img src="images/blockchain_header.png" alt="drawing" width="600"/>

## Overview
For this challenge I assume the role of a fintech engineer working in the decentralized finance team for a top bank. The challenge is to build a blockchain-based ledger system with user-friendly web interface that will allow partner banks to send and receive money and verify the integrity of the data in the ledger. 

## Libraries and Dependencies
The [Python File](pychain.py) loads the following libraries and dependencies.

```python
# Imports
import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import datetime as datetime
import pandas as pd
import hashlib
```

## End-user Instructions
The application will require a development environment that has the following installed:
* Python 3.10
* Pandas
* Streamlit

Once the file has been relocated to your local machine, the application can be run from a terminal window within your IDE (Visual Studio was used to develop the application) from the correct development environment containing the above, by typing the following command. 

```python
streamlit run pychain.py
```
## Application Results
Below are screenshots for the web application using Streamlit technology.

### Web application showing blockchain with multiple blocks
<img src="images/streamlit1.png" alt="drawing" width="700"/>

### Web application showing the chain validation
<small>(see `True` in bottom center of screenshot)</small>
<img src="images/streamlit2_validated.png" alt="drawing" width="700"/>

### Chain validation zoomed in
<img src="images/streamlit3_validated_closeup.png" alt="drawing" width="600"/>

### Close-up of expanded block inspector pulldown
<img src="images/streamlit4_blockinspector_pulldown.png" alt="drawing" width="500"/>

## Sources
The following sources were consulted in the completion of this project. 

* [pandas.Pydata.org API Reference](https://pandas.pydata.org/docs/reference/index.html)
* [Streamlit Documentation](https://docs.streamlit.io/)
* UCB FinTech Bootcamp instructor-led coding exercises

## License
[MIT License](LICENSE)