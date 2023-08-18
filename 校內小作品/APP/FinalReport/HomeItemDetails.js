import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, FlatList } from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
// import HomeItemDetailsItem from './HomeItemDetailsItem.js';
// import EpmtyItem from './epmtyItem'
// import HomeMapItem from './HomeMapItem';

function HomeItemDetails(props) {
  const { points, tlists } = props;
  const [k, setk] = useState(0);
  useEffect(() => {
    console.log(k)
  }, [k])
  return (
    <LinearGradient
      colors={['#66B3FF', '#ACD6FF', '#ECF5FF']}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={styles.container}
    >
      <View>
        <View>
          <View style={styles.contentTitle}>
            <Text style={styles.titleText}>{points.properties.name}</Text>
            {/* <Text style={styles.dateText}>{points.date}</Text> */}
          </View>
          <View style={styles.buttonContent}>
            <TouchableOpacity style={styles.buttonView_1} onPress={() => {
              setk(0);

            }}>
              <Text style={styles.buttonText_1}>行程表</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonView_1} onPress={() => {
              setk(1);

            }} >
              <Text style={styles.buttonText_1}>地圖</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.buttonContent}>
            <TouchableOpacity style={styles.buttonView_2}>
              <Text style={styles.buttonText_2}>Previous Day</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonView_2}>
              <Text style={styles.buttonText_2}>Next Day</Text>
            </TouchableOpacity>
          </View>
          {/* <View>
            {k == 0 ? tlists.map((tlist) => {
              return (
                <HomeItemDetailsItem key={tlist.id} id={tlist.id} />
              )
            }) : <MapTest />}

          </View> */}
        </View>
        {/* {console.log(points)} */}
      </View>
    </LinearGradient>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentTitle: {
    alignItems: 'center',
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  buttonView_1: {
    backgroundColor: '#F0F0F0',
    paddingHorizontal: 80, //tag區塊左右內距大小
    paddingVertical: 10, //tag區塊上下內距大小
    borderRadius: 2,
  },
  buttonView_2: {
    backgroundColor: '#000093',
    paddingHorizontal: 10, //tag區塊左右內距大小
    paddingVertical: 10, //tag區塊上下內距大小
    borderRadius: 2,
  },
  buttonText_1: {
    fontSize: 20,
    fontWeight: '200',
  },
  buttonText_2: {
    color: 'white',
    fontSize: 14,
  },
  titleText: {
    fontSize: 50,
    textShadowOffset: { width: 2, height: 2 },
    textShadowColor: 'gray',
    textShadowRadius: 2,
  },
  dateText: {
    fontSize: 14,
    fontWeight: 'bold',
    paddingVertical: 10,
    paddingLeft: 220,
  },
  content: {
    marginHorizontal: 10,
  }
})

export default HomeItemDetails;