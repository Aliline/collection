import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

class Member extends React.Component {
  render() {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>敬請期待</Text>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex:1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontWeight: 'bold',
    fontSize: 50,
  }
})

export default Member;
