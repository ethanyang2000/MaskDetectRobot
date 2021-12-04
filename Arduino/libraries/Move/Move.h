class Move{
public:
    Move(int In_PWML = 5, int In_CRLL = 6, int In_SLPL = 7, int In_PWMR = 2, int In_CRLR = 3, int In_SLPR = 4);
    Move(const Move& In_Move);
    ~Move();
    void turnleft();
    void turnright();
    void sleep();
    void fl();
    void fr();
    void bl();
    void br();
    void forwardright();
    void forwardleft();
    void backwardleft();
    void backwardright();
    void InitMove();

protected:
    int PWML, PWMR;
    int CRLL, CRLR;
    int SLPL, SLPR;
};

