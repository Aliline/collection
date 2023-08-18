import React from 'react';
import { StyleSheet, TouchableOpacity, Image, View, Text } from 'react-native';

const LocationItem = (props) => {
  const { location, onPress } = props;

  return (
    <TouchableOpacity onPress={() => onPress(location.id)} style={styles.locationItem}>
      <Image style={styles.image} source={{ uri: location.properties.photo }} />
      <View style={styles.locationContent}>
        <Text style={styles.locationName}>{location.properties.name}</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  locationList: {
    flex: 1,
  },
  locationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    borderBottomColor: '#DDD',
    borderBottomWidth: 1,
    paddingVertical: 10,
  },
  image: {
    width: 100,
    height: 100,
  },
  locationContent: {
    marginLeft: 10,
  },
  locationName: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'left',
    paddingVertical: 3,
  },
  locationPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F00',
    paddingBottom: 10,
  },
});

export default LocationItem;
