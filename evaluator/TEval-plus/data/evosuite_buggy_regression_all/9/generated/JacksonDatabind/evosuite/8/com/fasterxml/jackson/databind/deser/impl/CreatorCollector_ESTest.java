/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:39:37 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.CreatorCollector;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorCollector_ESTest extends CreatorCollector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addLongCreator((AnnotatedWithParams) null, true);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addLongCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      // Undeclared exception!
      try { 
        creatorCollector0.verifyNonDup((AnnotatedWithParams) null, 114);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 114
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null, false);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addDelegatingCreator((AnnotatedWithParams) null, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<SimpleModule> class0 = SimpleModule.class;
      ObjectReader objectReader0 = objectMapper0.reader((Class<?>) class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, (CreatorProperty[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addStringCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null, true);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(0);
      boolean boolean0 = creatorCollector_Vanilla0.canCreateUsingDefault();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(0);
      boolean boolean0 = creatorCollector_Vanilla0.canInstantiate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, false, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      Class<ObjectWriter> class0 = ObjectWriter.class;
      Class<TypeIdResolver> class1 = TypeIdResolver.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, classNameIdResolver0, (String) null, false, class1);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, (PropertyName) null, asWrapperTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 65599, class0, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[5];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      creatorPropertyArray0[2] = creatorProperty0;
      creatorPropertyArray0[3] = creatorProperty0;
      creatorPropertyArray0[4] = creatorProperty0;
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, false, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[2];
      String string0 = "_@ <K>ND3gO\\uzlJo@";
      PropertyName propertyName0 = new PropertyName(string0);
      Class<Method> class0 = Method.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, simpleType0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(collectionLikeType0, (TypeIdResolver) null, string0, true, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, class0, annotationMap0, 5036);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionLikeType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 5036, annotatedParameter0, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, false, creatorPropertyArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      Class<Field> class0 = Field.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, classNameIdResolver0, "dePgae", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 44, (Object) null, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[6];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, false, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Duplicate creator property \"\" (index 0 vs 1)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<AnnotatedMethod> class1 = AnnotatedMethod.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, mapType0, (AnnotationMap) null, (-5299));
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      assertEquals((-5299), annotatedParameter0.getIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature mapperFeature0 = MapperFeature.CAN_OVERRIDE_ACCESS_MODIFIERS;
      objectMapper0.configure(mapperFeature0, false);
      Class<SimpleModule> class0 = SimpleModule.class;
      ObjectReader objectReader0 = objectMapper0.reader((Class<?>) class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorCollector_Vanilla0.createFromInt(defaultDeserializationContext_Impl0, 3);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not instantiate value of type java.util.ArrayList from Integer number (3, int)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createFromLong((DeserializationContext) null, 3);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.LinkedHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(0);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ArrayList arrayList0 = (ArrayList)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertTrue(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      LinkedHashMap linkedHashMap0 = (LinkedHashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, linkedHashMap0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      HashMap hashMap0 = (HashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3263);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createUsingDefault((DeserializationContext) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown type 3263
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector$Vanilla", e);
      }
  }
}