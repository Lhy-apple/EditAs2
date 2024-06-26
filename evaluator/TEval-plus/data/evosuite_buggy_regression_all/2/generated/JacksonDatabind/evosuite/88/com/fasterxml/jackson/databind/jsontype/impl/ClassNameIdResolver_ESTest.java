/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:04:58 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.FormatSchema;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DatabindContext;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.EnumMap;
import java.util.EnumSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ClassNameIdResolver_ESTest extends ClassNameIdResolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectReader objectReader0 = objectMapper0.reader((FormatSchema) null);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<Integer> class0 = Integer.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      Class<String> class1 = String.class;
      classNameIdResolver0.registerSubtype(class1, "JSON");
      assertEquals("class name used as type id", classNameIdResolver0.getDescForKnownTypeIds());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      String string0 = classNameIdResolver0.idFromValueAndType(class0, class0);
      assertEquals("java.util.EnumSet", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      JsonTypeInfo.Id jsonTypeInfo_Id0 = classNameIdResolver0.getMechanism();
      assertEquals("@class", jsonTypeInfo_Id0.getDefaultPropertyName());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      // Undeclared exception!
      try { 
        classNameIdResolver0.typeFromId((DatabindContext) null, "java.utiljava.util.EnumMap");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      String string0 = classNameIdResolver0.getDescForKnownTypeIds();
      assertEquals("class name used as type id", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      String string0 = classNameIdResolver0.idFromValue(collectionType0);
      assertEquals("com.fasterxml.jackson.databind.type.CollectionType", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      TypeFactory typeFactory0 = serializerProvider0.getTypeFactory();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      JavaType javaType0 = classNameIdResolver0._typeFromId("java.utilcom.fasterxml.jackson.databind.JsonDeserializer", serializerProvider0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      // Undeclared exception!
      try { 
        classNameIdResolver0._typeFromId("?KJmHK<GR`7yx7%v1", serializerProvider0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '?KJmHK<GR`7yx7%v1' (remaining: '<GR`7yx7%v1'): Can not locate class '?KJmHK', problem: Class '?KJmHK.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      String string0 = classNameIdResolver0._idFrom((Object) null, class0, typeFactory0);
      assertEquals("com.fasterxml.jackson.databind.AnnotationIntrospector$ReferenceProperty$Type", string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      EnumMap<DeserializationFeature, AnnotationIntrospector.ReferenceProperty.Type> enumMap0 = new EnumMap<DeserializationFeature, AnnotationIntrospector.ReferenceProperty.Type>(class1);
      EnumMap<DeserializationFeature, Object> enumMap1 = new EnumMap<DeserializationFeature, Object>(enumMap0);
      String string0 = minimalClassNameIdResolver0._idFrom(enumMap1, class0, typeFactory0);
      assertEquals("java.util.EnumMap<com.fasterxml.jackson.databind.DeserializationFeature,java.lang.Object>", string0);
  }
}
