/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:25:36 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.CreatorCollector;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.testdata.EvoSuiteFile;
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
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addLongCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      AnnotatedWithParams annotatedWithParams0 = creatorCollector0.verifyNonDup((AnnotatedWithParams) null, 1);
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null, true);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      creatorCollector0.setDefaultCreator(annotatedConstructor0);
      boolean boolean0 = creatorCollector0.hasDefaultCreator();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addStringCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null, true);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(678);
      boolean boolean0 = creatorCollector_Vanilla0.canCreateUsingDefault();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      boolean boolean0 = creatorCollector_Vanilla0.canInstantiate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forCreation((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, true, creatorPropertyArray0);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertTrue(valueInstantiator0.canCreateFromObjectWith());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertFalse(valueInstantiator0.canCreateFromInt());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      creatorCollector0.addDelegatingCreator((AnnotatedWithParams) annotatedConstructor0, false, (CreatorProperty[]) null);
      // Undeclared exception!
      try { 
        creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.BasicBeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addDelegatingCreator((AnnotatedWithParams) annotatedConstructor0, true, creatorPropertyArray0);
      // Undeclared exception!
      try { 
        creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.BasicBeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[6];
      PropertyName propertyName0 = new PropertyName("", "");
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      creatorCollector0.addDelegatingCreator((AnnotatedWithParams) annotatedConstructor0, creatorPropertyArray0);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 92, annotationMap0, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.BasicBeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class1 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class1, class0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, collectionType0, (AnnotatedClass) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertFalse(valueInstantiator0.canCreateUsingDelegate());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      Class<HashMap> class1 = HashMap.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<Method> class2 = Method.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class2, simpleType0, simpleType0);
      MapType mapType0 = MapType.construct(class0, simpleType0, mapLikeType0);
      MapType mapType1 = typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) mapType0, (JavaType) mapLikeType0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, mapType1, (AnnotatedClass) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertFalse(valueInstantiator0.canCreateFromObjectWith());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      PropertyName propertyName0 = new PropertyName("", "");
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 7, annotationMap0, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[3];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      creatorPropertyArray0[2] = creatorProperty0;
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, false, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[2];
      PropertyName propertyName0 = new PropertyName("TG", "TG");
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 120, annotationMap0, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, false, creatorPropertyArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      PropertyName propertyName0 = new PropertyName("", "");
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 92, (Object) null, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[4];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, true, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Duplicate creator property \"\" (index 0 vs 1)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, mapLikeType0, annotationMap0, (-419));
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      assertEquals("", annotatedParameter0.getName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      boolean boolean0 = creatorCollector0.hasDefaultCreator();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forCreation((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, true, creatorPropertyArray0);
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations((MapperConfig<?>) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[1];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, creatorPropertyArray0);
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, false, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting property-based creators: already had explicitly marked [constructor for java.lang.String, annotations: [null]], encountered [constructor for java.lang.String, annotations: [null]]
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forDeserialization((DeserializationConfig) null, mapLikeType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[19];
      creatorCollector0.addDelegatingCreator((AnnotatedWithParams) annotatedConstructor0, true, creatorPropertyArray0);
      // Undeclared exception!
      try { 
        creatorCollector0.addDelegatingCreator((AnnotatedWithParams) annotatedConstructor0, true, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting delegate creators: already had explicitly marked [constructor for java.lang.String, annotations: [null]], encountered [constructor for java.lang.String, annotations: [null]]
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.ArrayList", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla((-65));
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.LinkedHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.HashMap", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      ArrayList arrayList0 = (ArrayList)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, arrayList0.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      LinkedHashMap linkedHashMap0 = (LinkedHashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertTrue(linkedHashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      HashMap hashMap0 = (HashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3127);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createUsingDefault((DeserializationContext) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown type 3127
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector$Vanilla", e);
      }
  }
}
