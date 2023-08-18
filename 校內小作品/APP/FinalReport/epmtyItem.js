import React from 'react';
import { View, Text,StyleSheet } from 'react-native';

class EpmtyItem extends React.Component {
  render() {
    return (
      <View style={myStyle.epmtyWarp}>
        <Text style={myStyle.epmtyItem}>還沒有內容喔!</Text>
        <Text style={myStyle.epmtyItem}>現在就去加入吧!</Text>
      </View>
    );

  }

}
const myStyle = StyleSheet.create({
    epmtyWarp:{
        alignItems : 'center',
        justifyContent : 'center'
    },
    epmtyItem:{
        textAlign :'center',
        fontSize : 32
      },
})
export default EpmtyItem;