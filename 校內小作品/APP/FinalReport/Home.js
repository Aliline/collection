import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import { Actions } from 'react-native-router-flux';
import point from './point.json';
// import point from './point.js';  // 測試檔
import HomeList from './HomeList.js';

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      points: point.features,
      tlists: props.tlists,
    }
  }
  
  handleRedirectHomeItemDetails = (id) => {
    const { points ,tlists} = this.state;
    const point = points.find((point) => point.id === id);

    // 跳轉至餐點詳細頁面時將底部的 Tab 隱藏
    Actions.push('HomeItemDetails', { points: point, hideTabBar: true ,tlists : tlists});
  };

  render() {
    const { points } = this.state;
    const { handleRedirectHomeItemDetails } = this;

    return (
      <LinearGradient
        colors={['#66B3FF', '#ACD6FF', '#ECF5FF']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.containt}
      >
        <View>
          <View style={styles.date}>
            <Text style={styles.dateYearText}>{new Date().getFullYear()}</Text>
            <Text style={styles.dateMonthText}>{new Date().getMonth()+1}</Text>
            <Text style={styles.bar}>/</Text>
            <Text style={styles.dateDateText}>{new Date().getDate()}</Text>
            <Text style={styles.dateTime}>{new Date().toLocaleTimeString()}</Text>
          </View>
          <ScrollView>
            <HomeList points={points} onPress={handleRedirectHomeItemDetails} />
          </ScrollView>
        </View>
      </LinearGradient>
    )
  }
}

const styles = StyleSheet.create({
  containt: {
    flex: 1,
  },
  date: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 150,
  },
  dateYearText: {
    transform: [{ rotate: '90deg' }],
    color: '#FDFFFF',
    fontSize: 50,
    fontWeight: 'bold',
  },
  dateMonthText: {
    color: '#FDFFFF',
    fontSize: 50,
    paddingBottom: 50,
    fontWeight: 'bold',
  },
  bar: {
    color: '#FDFFFF',
    fontSize: 120,
    margin: 20,
    paddingBottom: 20,
  },
  dateDateText: {
    color: '#FDFFFF',
    fontSize: 50,
    fontWeight: 'bold',
    paddingTop: 30,
  },
  dateTime: {
    color: '#FDFFFF',
    fontSize: 30,
    paddingLeft: 20,
    paddingBottom: 60,
  }
})

export default Home;