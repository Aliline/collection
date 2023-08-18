import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import HomeItemDetailsItem from './HomeItemDetailsItem.js';
import HomeItems from './HomeItems.js';

function HomeList(props) {
  const { points, onPress } = props;

  return (
    <ScrollView style={styles.content}>
      {points.map((point) => {
        return (
          <HomeItems key={point.id} point={point} onPress={onPress} />
        )
      })}
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  content: {
    marginHorizontal: 10,
  }
})

export default HomeList;