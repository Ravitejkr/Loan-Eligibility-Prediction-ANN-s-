def replacer(df):
    import pandas as pd
    Q = pd.DataFrame(df.isna().sum(),columns=["CT"])
    W = Q[Q.CT > 0].index
    for i in W:
        if(df[i].dtypes == "object"):
            x = df[i].mode()[0]
            df[i] = df[i].fillna(x)
        else:
            x = df[i].mean()
            df[i] = df[i].fillna(x)
            
            
def data_prep(X):
    from automation import replacer
    replacer(X)
    cat = []
    con = []
    for i in X.columns:
        if(X[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)
            
    import pandas as pd
    X1 = pd.get_dummies(X[cat])
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X2 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
    X3 = X2.join(X1)
    return X3
            
def model_lm(df,ycol,xcols):
    from automation import replacer
    replacer(df)
    import pandas as pd
    Y = df[ycol]
    X = df[xcols]
    cat = []
    con = []
    for i in X.columns:
        if(X[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
    X2 = pd.get_dummies(X[cat])
    X = X1.join(X2)
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=21)
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    model = lm.fit(xtrain,ytrain)
    from sklearn.metrics import mean_squared_error
    pred_tr = model.predict(xtrain)
    tr_err = round(mean_squared_error(ytrain,pred_tr),3)
    pred_ts = model.predict(xtest)
    ts_err = round(mean_squared_error(ytest,pred_ts),3)
    print("=================\n")
    print(ycol[0],"~",xcols)
    print("Training Error: ",tr_err)
    print("Testing Error: ",ts_err)
    if(tr_err < ts_err):
        print("Overfitting")
    print("\n=================\n")
    
    
    
    
def model_builder(mobj):
    from sklearn.metrics import accuracy_score
    model = mobj.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    return tr_acc,ts_acc    