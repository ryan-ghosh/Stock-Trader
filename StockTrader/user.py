# import buyModel
# import sellmodel

class User:
    positions = {}

    def update(self):
        ## this will use TA library to call and pass info to NN
        pass


    def add_position(self, stock):
        self.positions[stock] = True
    
    def sell_position(self, stock):
        self.positions.pop(stock)

    def show_positions(self):
        if self.positions:
            holdings = []
            for key, value in self.positions.items():
                holdings.append(key)
            print(holdings)
        else:
            print('No positions currently held')
    
if __name__ =="__main__":
    Ryan = User()
    Ryan.add_position('TSLA')
    Ryan.show_positions()
    Ryan.sell_position('TSLA')
    Ryan.show_positions()
    