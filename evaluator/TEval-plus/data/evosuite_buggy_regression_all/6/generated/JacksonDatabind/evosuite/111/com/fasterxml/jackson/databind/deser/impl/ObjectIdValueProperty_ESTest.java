/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:31:11 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<String> class0 = String.class;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator(class0, 663);
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withNullProvider((NullValueProvider) null);
      assertFalse(settableBeanProperty0.hasViews());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set(propertyMetadata0, (Object) null);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<String> class0 = String.class;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator(class0, 663);
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withName(objectIdReader0.propertyName);
      assertNotSame(objectIdValueProperty0, settableBeanProperty0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "s;_>9[tQH'9$J");
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<ShortNode> class0 = ShortNode.class;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator(class0, 0);
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Integer integer0 = new Integer(3588);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser((byte[]) null, 0, 0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, integer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<Annotation> jsonDeserializer0 = (JsonDeserializer<Annotation>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withValueDeserializer(jsonDeserializer0);
      assertTrue(settableBeanProperty0.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertSame(settableBeanProperty0, objectIdValueProperty0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(javaType0, javaType0, javaType0);
      PropertyName propertyName0 = new PropertyName("qcYLj&}H)TEnLARUhI2", "qcYLj&}H)TEnLARUhI2");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(mapLikeType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdReader objectIdReader1 = new ObjectIdReader(javaType0, propertyName0, objectIdReader0.generator, (JsonDeserializer<?>) null, objectIdValueProperty0, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty1.setAndReturn("qcYLj&}H)TEnLARUhI2", javaType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}