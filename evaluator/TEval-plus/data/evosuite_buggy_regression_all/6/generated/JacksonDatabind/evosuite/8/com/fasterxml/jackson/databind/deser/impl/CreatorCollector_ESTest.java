/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:20:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.type.ResolvedType;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
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
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
        creatorCollector0.verifyNonDup((AnnotatedWithParams) null, (-2400));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2400
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<Integer> class1 = Integer.class;
      Class<JsonDeserializer> class2 = JsonDeserializer.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class2);
      MappingIterator<HashMap<Object, AnnotatedConstructor>> mappingIterator0 = objectMapper0.readValues(jsonParser0, (ResolvedType) mapType0);
      assertNotNull(mappingIterator0);
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
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[0];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, creatorPropertyArray0);
      assertEquals(0, creatorPropertyArray0.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
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
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null, false);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(746);
      boolean boolean0 = creatorCollector_Vanilla0.canCreateUsingDefault();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(746);
      boolean boolean0 = creatorCollector_Vanilla0.canInstantiate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Method> class0 = Method.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectReader objectReader0 = objectMapper0.reader(jsonNodeFactory0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<LinkedHashMap> class1 = LinkedHashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class1);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, mapType0, annotatedClass0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertTrue(valueInstantiator0.canInstantiate());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<String> class0 = String.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      Class<Object> class1 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerWithView((Class<?>) class1);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<HashMap> class2 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class2);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, mapType0, annotatedClass0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator((DeserializationConfig) null);
      assertFalse(valueInstantiator0.canCreateFromObjectWith());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      Class<String> class1 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      Class<JsonToken> class2 = JsonToken.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class2, annotationIntrospector1, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = new AnnotationMap();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, mapType0, annotatedClass0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      PropertyName propertyName0 = new PropertyName("java.lang.Object", "java.lang.Object");
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[3];
      PropertyName propertyName1 = PropertyName.USE_DEFAULT;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      Class<Method> class3 = Method.class;
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(mapType0, classNameIdResolver0, "java.lang.Object", true, class3);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName1, mapType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, (-2985), annotationMap0, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      creatorPropertyArray0[2] = creatorPropertyArray0[1];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, true, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<Integer> class1 = Integer.TYPE;
      Integer integer0 = new Integer(14);
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, false);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[7];
      PropertyName propertyName0 = new PropertyName("");
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, classNameIdResolver0, "", true, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, class0, annotationMap0, 14);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(false, "", integer0, (String) null);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 98, (Object) null, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, true, creatorPropertyArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      Class<String> class1 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      Class<JsonToken> class2 = JsonToken.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class2, annotationIntrospector1, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = new AnnotationMap();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, mapType0, annotatedClass0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, true);
      PropertyName propertyName0 = new PropertyName("java.lang.Object", "java.lang.Object");
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[3];
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      Class<Method> class3 = Method.class;
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(mapType0, classNameIdResolver0, "java.lang.Object", true, class3);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 2, annotationMap0, propertyMetadata0);
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, true, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Duplicate creator property \"java.lang.Object\" (index 0 vs 1)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, class0, annotationMap0, (-457));
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      assertEquals((-457), annotatedParameter0.getIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, annotatedClass0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, false);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) annotatedConstructor0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.ArrayList", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla((-9));
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.LinkedHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.HashMap", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ArrayList arrayList0 = (ArrayList)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, arrayList0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      HashMap hashMap0 = (HashMap)creatorCollector_Vanilla0.createUsingDefault((DeserializationContext) null);
      assertTrue(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      LinkedHashMap linkedHashMap0 = (LinkedHashMap)creatorCollector_Vanilla0.createUsingDefault(deserializationContext0);
      assertTrue(linkedHashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2312);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createUsingDefault((DeserializationContext) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown type 2312
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector$Vanilla", e);
      }
  }
}