/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:00:25 GMT 2023
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
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.node.IntNode;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      Class<ObjectReader> class0 = ObjectReader.class;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator(class0, 0);
      Integer integer0 = objectIdGenerators_IntSequenceGenerator0.generateId((Object) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(true, "Can not set virtual property '", integer0, "");
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      IntNode intNode0 = new IntNode(0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set((Object) null, intNode0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Integer integer0 = new Integer((-65));
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(integer0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerators_IntSequenceGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      objectIdValueProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, (Object) null);
      assertFalse(jsonParser0.hasCurrentToken());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withName(objectIdReader0.propertyName);
      assertEquals("", objectIdValueProperty1.getName());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("y :b", "y :b");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdValueProperty0, "y :b");
      assertFalse(objectIdValueProperty1.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<Integer> jsonDeserializer1 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer1).getNullValue();
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer(jsonDeserializer1);
      assertEquals("", objectIdValueProperty1.getName());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<JsonParser.Feature> class0 = JsonParser.Feature.class;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator(class0, 1505);
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerators_IntSequenceGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<ObjectIdGenerators.IntSequenceGenerator> jsonDeserializer1 = (JsonDeserializer<ObjectIdGenerators.IntSequenceGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader1 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerators_IntSequenceGenerator0, jsonDeserializer1, objectIdValueProperty0, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      IntNode intNode0 = new IntNode(1648);
      // Undeclared exception!
      try { 
        objectIdValueProperty1.setAndReturn(intNode0, (Object) null);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}
