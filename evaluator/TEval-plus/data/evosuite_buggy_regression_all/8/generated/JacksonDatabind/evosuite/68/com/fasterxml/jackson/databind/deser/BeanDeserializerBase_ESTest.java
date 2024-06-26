/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:14:51 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.ext.NioPathDeserializer;
import com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator;
import java.util.LinkedHashSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerBase_ESTest extends BeanDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LinkedHashSet<SettableBeanProperty> linkedHashSet0 = new LinkedHashSet<SettableBeanProperty>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) linkedHashSet0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not deserialize instance of com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator out of START_ARRAY token
         //  at [Source: java.lang.String@0000000146; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[4];
      MapperFeature mapperFeature0 = MapperFeature.AUTO_DETECT_FIELDS;
      mapperFeatureArray0[0] = mapperFeature0;
      mapperFeatureArray0[1] = mapperFeatureArray0[0];
      mapperFeatureArray0[2] = mapperFeatureArray0[0];
      MapperFeature mapperFeature1 = MapperFeature.DEFAULT_VIEW_INCLUSION;
      mapperFeatureArray0[3] = mapperFeature1;
      objectMapper0.disable(mapperFeatureArray0);
      Class<RuntimeException> class0 = RuntimeException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_CONCRETE_AND_ARRAYS;
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_OBJECT;
      objectMapper0.enableDefaultTyping(objectMapper_DefaultTyping0, jsonTypeInfo_As0);
      Class<NioPathDeserializer> class0 = NioPathDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<CreatorProperty> class0 = CreatorProperty.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_FINAL;
      objectMapper0.enableDefaultTyping(objectMapper_DefaultTyping0);
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) objectMapper_DefaultTyping0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping not subtype of [simple type, class com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) objectIdGenerators_StringIdGenerator0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator: no suitable constructor found, can not deserialize from Object value (missing default constructor or creator, or perhaps need to add/enable type information?)
         //  at [Source: java.lang.String@0000001082; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Long long0 = new Long((-2924L));
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) long0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator: no long/Long-argument constructor/factory method to deserialize from Number value (-2924)
         //  at [Source: java.lang.String@0000000422; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Integer integer0 = new Integer(0);
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) integer0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator: no int/Int-argument constructor/factory method to deserialize from Number value (0)
         //  at [Source: java.lang.String@0000000422; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<PropertyBasedObjectIdGenerator> class0 = PropertyBasedObjectIdGenerator.class;
      try { 
        objectMapper0.convertValue((Object) class0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator: no String-argument constructor/factory method to deserialize from String value ('com.fasterxml.jackson.databind.ser.impl.PropertyBasedObjectIdGenerator')
         //  at [Source: java.lang.String@0000001041; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }
}
