/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:43:57 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, ";U>tFw//ZP`", true, (JavaType) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      String string0 = objectMapper0.writeValueAsString(asPropertyTypeDeserializer0);
      assertEquals("{\"typeInclusion\":\"PROPERTY\",\"typeIdResolver\":null,\"defaultImpl\":null,\"propertyName\":\";U>tFw//ZP`\"}", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "*r?OJ:w:::<qMHr", true, (JavaType) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleMissingTypeId(deserializationContext0, "*r?OJ:w:::<qMHr");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "vLaA-VFf", false, (JavaType) null);
      String string0 = asWrapperTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer; base-type:null; id-resolver: null]", string0);
      assertEquals("vLaA-VFf", asWrapperTypeDeserializer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, ";F;|oQ0(0.9", true, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser(";F;|oQ0(0.9");
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "", true, (JavaType) null);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<List> class0 = List.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class1);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(mapType0, classNameIdResolver0, "eGx3dPQSw1!Wn7bX-@", false, mapType0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer1 = new AsWrapperTypeDeserializer(asWrapperTypeDeserializer0, beanProperty_Bogus0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer1._handleUnknownTypeId((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<List> class0 = List.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, minimalClassNameIdResolver0, "Internal error: total len assumed to be ", true, collectionType0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._findDefaultImplDeserializer((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, ";F;|oQ0(0.9", true, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser(";F;|oQ0(0.9");
      asWrapperTypeDeserializer0._deserializeWithNativeTypeId(jsonParser0, defaultDeserializationContext_Impl0, (Object) null);
      assertEquals(";F;|oQ0(0.9", asWrapperTypeDeserializer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "5hZE#NP", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<List> class0 = List.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class1);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(mapType0, classNameIdResolver0, "eGx3dPQSw1!Wn7bX-@", false, mapType0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}