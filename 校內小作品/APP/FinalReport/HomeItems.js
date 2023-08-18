import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

function HomeItems(props) {
  const { point, onPress } = props;  

  return (
    <TouchableOpacity onPress={() => onPress(point.id)}>
      <View style={styles.content}>
        {/* <Text style={styles.text}>{point.date}</Text> */}
        <Text style={styles.text}>{point.properties.name}</Text>
        {/* <Text style={styles.text}>{point.content}</Text> */}
      </View>
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  content: {
    justifyContent: 'space-between', // 區塊貼齊左右兩邊
    backgroundColor: 'white',
    borderRadius: 2, // 邊框圓角
    marginVertical: 5, // 區塊上下垂直外距大小
    padding: 10,
    elevation: 5, // 陰影深淺
  },
  text: {
    fontSize: 14,
  }
})

export default HomeItems;