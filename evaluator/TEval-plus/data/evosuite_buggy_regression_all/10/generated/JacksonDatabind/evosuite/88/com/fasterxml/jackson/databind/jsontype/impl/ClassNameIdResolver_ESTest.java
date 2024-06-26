/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:40:24 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.EnumMap;
import java.util.EnumSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ClassNameIdResolver_ESTest extends ClassNameIdResolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumMap> class0 = EnumMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(mapType0, typeFactory0);
      Class<Integer> class1 = Integer.class;
      minimalClassNameIdResolver0.registerSubtype(class1, "java.lang.String");
      assertEquals("class name used as type id", minimalClassNameIdResolver0.getDescForKnownTypeIds());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      String string0 = minimalClassNameIdResolver0.idFromBaseType();
      assertEquals("java.util.EnumSet", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(referenceType0, typeFactory0);
      JsonTypeInfo.Id jsonTypeInfo_Id0 = classNameIdResolver0.getMechanism();
      assertEquals(JsonTypeInfo.Id.CLASS, jsonTypeInfo_Id0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        minimalClassNameIdResolver0.typeFromId(defaultSerializerProvider_Impl0, "_");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumMap> class0 = EnumMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(mapType0, typeFactory0);
      String string0 = minimalClassNameIdResolver0.getDescForKnownTypeIds();
      assertEquals("class name used as type id", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(resolvedRecursiveType0, (TypeFactory) null);
      String string0 = classNameIdResolver0.idFromValue(class0);
      assertEquals("java.lang.Class", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      JavaType javaType0 = minimalClassNameIdResolver0._typeFromId("x[l?JN", serializerProvider0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      // Undeclared exception!
      try { 
        minimalClassNameIdResolver0._typeFromId(".Co$je<ctions$", serializerProvider0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'java.util.Co$je<ctions$' (remaining: '<ctions$'): Can not locate class 'java.util.Co$je', problem: java.util.Co$je
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      String string0 = minimalClassNameIdResolver0._idFrom(typeFactory0, class1, typeFactory0);
      assertEquals("com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping", string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      EnumMap<ObjectMapper.DefaultTyping, MockFileInputStream> enumMap0 = new EnumMap<ObjectMapper.DefaultTyping, MockFileInputStream>(class1);
      String string0 = minimalClassNameIdResolver0._idFrom(enumMap0, class0, typeFactory0);
      assertEquals("java.util.EnumMap<com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping,java.lang.Object>", string0);
  }
}
