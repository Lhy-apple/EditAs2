/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:26:32 GMT 2023
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
import com.fasterxml.jackson.core.JsonGenerator;
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
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(true, "", integer0, "");
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
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
  public void test2()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertEquals((-1), objectIdValueProperty1.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("");
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, jsonFactory0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
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
      assertFalse(objectIdValueProperty1.isRequired());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdValueProperty0, (String) null);
      assertFalse(objectIdValueProperty1.hasViews());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Integer integer0 = new Integer(1505);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(integer0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<Object> class0 = Object.class;
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.AUTO_CLOSE_SOURCE;
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = new ObjectIdGenerator.IdKey(class0, class0, jsonParser_Feature0);
      ObjectIdGenerator<JsonParser.Feature> objectIdGenerator0 = (ObjectIdGenerator<JsonParser.Feature>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn((ObjectIdGenerator.IdKey) null).when(objectIdGenerator0).key(any());
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ReadableObjectId", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdGenerator<JsonParser.Feature> objectIdGenerator0 = (ObjectIdGenerator<JsonParser.Feature>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.FLUSH_PASSED_TO_STREAM;
      JsonDeserializer<IntNode> jsonDeserializer1 = (JsonDeserializer<IntNode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader1 = new ObjectIdReader((JavaType) null, objectIdReader0.propertyName, objectIdReader0.generator, jsonDeserializer1, objectIdValueProperty0, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty1.set(jsonFactory0, jsonGenerator_Feature0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}