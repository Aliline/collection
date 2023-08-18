/* eslint-disable prettier/prettier */
import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';

export default class ScheduleList extends React.Component {
  render() {
    return (
      <View>
        <TouchableOpacity style={[myStyle.content, myStyle.unCompletedBorder]}>
          {/* 大區塊一 */}
          <View style={myStyle.imageContent}>
            <Image source={{ uri: this.props.todo.properties.photo }} style={myStyle.image} />
          </View>

          {/*大區塊二*/}
          <View style={myStyle.tourContent}>
            {/* 中區塊一：title、subTitle */}
            <View>
              {/* 根據完成狀態顯示不同的標題樣式 */}
              <Text style={myStyle.attractionName}>{this.props.todo.properties.name}</Text>
            </View>

            {/*中區塊二：Name、刪除*/}
            <View>
              <TouchableOpacity style={myStyle.updateView} onPress={() => this.props.onPress2(this.props.tlist.id)}>
                <Text style={myStyle.tagText}>刪除</Text>
              </TouchableOpacity>
            </View>
          </View>
        </TouchableOpacity>
      </View>
    );
  }
}

const myStyle = StyleSheet.create({
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    borderLeftWidth: 10,
    borderRadius: 8,
    marginVertical: 10,
    padding: 10,
    elevation: 10,
  },
  unCompletedBorder: {
    borderLeftColor: '#3366FF',
  },
  imageContent: {
    flex: 0.3, //設定大區塊
  },
  image: {
    width: 70,
    height: 70,
    borderRadius: 35,
  },
  tourContent: {
    flex: 0.9, //設定大區塊
    flexDirection: 'row',
    justifyContent: 'space-between', //區塊貼齊左右兩邊
    alignItems: 'center',
  },
  attractionName: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  subTitle: {
    fontSize: 14,
    color: 'gray',
    paddingTop: 8, //新增subTitle區塊上下內距大小
  },
  //新增
  insertView: {
    alignSelf: 'flex-start', //tag標籤那一行的上面，這行有無好像沒差????
    backgroundColor: '#00F', //tag區塊背景顏色：藍色
    paddingHorizontal: 8, //tag區塊左右內距大小
    paddingVertical: 5, //tag區塊上下內距大小
    borderRadius: 5,
  },
  updateView: {
    alignSelf: 'flex-start', //tag標籤那一行的上面，這行有無好像沒差????
    backgroundColor: '#F00', //tag區塊背景顏色：藍色
    paddingHorizontal: 8, //tag區塊左右內距大小
    paddingVertical: 5, //tag區塊上下內距大小
    borderRadius: 5,
  },
  tagText: {
    color: 'white',
    fontSize: 14,
  },
});
