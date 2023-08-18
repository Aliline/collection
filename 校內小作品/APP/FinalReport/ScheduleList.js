/* eslint-disable prettier/prettier */
import React from 'react';
import { View, Text, ScrollView, StyleSheet, Alert, Button,FlatList } from 'react-native';
import { Actions } from 'react-native-router-flux';
import ScheduleItem from './ScheduleItem';
import EpmtyItem from './epmtyItem'
// import Gura from './Images/Gura.jpg';
// import Coli from './Images/Coli.jpg';
// import Marine from './Images/Marine.png';
import point from './point.json';


//同ch8 TodoList，只是移除TodoForm改到Router去呼叫、多了Action

export default class TodoList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      todos: point.features,
      tlists : props.tlists
    };
  }

  // handleToUpdate = (id) => {
  //   const { todos } = this.state;
  //   const { handleUpDate } = this.props;
  //   const todo = todos.find((todo) => id === todo.id);
  //   console.log(todo);
  //   Actions.push('UpdateForm', { todo: todo, handleUpDate: handleUpDate });
  // };

  // handlePress = (id) => {
  //   const newTodos = this.state.todos.map((todo) => {
  //     return todo.id === id ? { ...todo, completed: !todo.completed } : todo;
  //   });

  //   this.setState({
  //     todos: newTodos,
  //   });
  // };

  handleCreate = (id) => {
    Alert.alert(
      '新增',
      '是否新增',
      [
        {
          text: 'Cancel',
          onPress: () => {},
          style: 'cancel',
        },
        {
          text: 'OK',
          onPress: () => {
            /*新增方法(id)*/
          },
        },
      ],
      { cancelable: false },
    );
  };

  handleDelete = (id) => {
    Alert.alert(
      '刪除',
      '是否刪除',
      [
        {
          text: 'Cancel',
          onPress: () => {},
          style: 'cancel',
        },
        {
          text: 'OK',
          onPress: () => {
            /*刪除方法(id)*/
            let clone = JSON.parse(JSON.stringify(this.state.tlists));
            const index = clone.findIndex((obj) => obj.id === id);
            if (index !== -1) clone.splice(index, 1);
            this.setState({
              tlists : clone,
            });
          },
        },
      ],
      { cancelable: false },
    );
  };

  render() {
    const { stitle, startDate, endDate } = this.props;

    return (
      <View style={myStyle.container}>
        

        <ScrollView>
          
          <View>
            <View style={myStyle.tourItems}>
              {this.state.todos.map((todo) => (           
                this.state.tlists.map((tlist) =>{
                  if (tlist.id == todo.id) {
                      return(
                        <ScheduleItem
                        key={todo.id}
                        todo={todo}
                        tlist={tlist}
                        onPress1={this.handleCreate}
                        onPress2={this.handleDelete}
                      />
                      )
                  }
                  
                })

              ))}
              {this.state.tlists.length == 0 && <EpmtyItem/> } 
              

                
              
            </View>
          </View>
        </ScrollView>


        <Button title="完成" color="green" />
      </View>
    );
  }
}

const myStyle = StyleSheet.create({
  container: {
    flex: 1, // 分割畫面區塊
    backgroundColor: '#F4F4F4', // 版面背景顏色(灰色)
  },
  tourTitle: {
    textAlign: 'center', // 標題文字置中
    fontSize: 25, // 標題文字大小
    fontWeight: 'bold', // 標題文字粗細
    paddingVertical: 10, // 上下垂直內聚大小
  },
  //  又把todoItem還原了
  tourItems: {
    marginHorizontal: 10, // TodoItems 整個區塊的左右外距大小
  },
  epmtyItem:{
    textAlign :'center',
    fontSize : 32
  },
});
