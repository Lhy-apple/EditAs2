/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:43:48 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AsPropertyTypeDeserializer_ESTest extends AsPropertyTypeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, classNameIdResolver0, "JSON", false, (JavaType) null);
      TypeDeserializer typeDeserializer0 = asPropertyTypeDeserializer0.forProperty((BeanProperty) null);
      assertSame(typeDeserializer0, asPropertyTypeDeserializer0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_ARRAY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, classNameIdResolver0, "com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer", true, (JavaType) null, jsonTypeInfo_As0);
      PropertyName propertyName0 = PropertyName.construct("com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer");
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, asPropertyTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, (-3755), annotationMap0, (PropertyMetadata) null);
      assertFalse(creatorProperty0.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, (String) null, false, simpleType0, jsonTypeInfo_As0);
      JsonTypeInfo.As jsonTypeInfo_As1 = asPropertyTypeDeserializer0.getTypeInclusion();
      assertEquals(JsonTypeInfo.As.PROPERTY, jsonTypeInfo_As1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, classNameIdResolver0, "JSON", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = asPropertyTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, classNameIdResolver0, "JSON", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.createInstance(deserializationConfig0, jsonParser0, injectableValues_Std0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.TypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, classNameIdResolver0, "JSON", false, (JavaType) null);
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0, defaultDeserializationContext_Impl0);
      Object object0 = asPropertyTypeDeserializer0._deserializeTypedUsingDefaultImpl(jsonParser0, defaultDeserializationContext_Impl0, tokenBuffer0);
      assertNull(object0);
  }
}