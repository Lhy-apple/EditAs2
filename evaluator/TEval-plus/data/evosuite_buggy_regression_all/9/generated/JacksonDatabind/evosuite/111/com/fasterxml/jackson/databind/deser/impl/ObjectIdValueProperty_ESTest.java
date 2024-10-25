/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:00:30 GMT 2023
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
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.lang.annotation.Annotation;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(simpleType0, propertyName0, objectIdGenerators_StringIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) null, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withNullProvider((NullValueProvider) null);
      assertFalse(settableBeanProperty0.isVirtual());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(simpleType0, propertyName0, objectIdGenerators_StringIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set("Should not call set() on ObjectIdProperty that has no SettableBeanProperty", objectIdGenerators_StringIdGenerator0);
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
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) null, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withName(propertyName0);
      assertFalse(settableBeanProperty0.hasValueTypeDeserializer());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(simpleType0, propertyName0, objectIdGenerators_StringIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class1 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class1);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(simpleType0, propertyName0, objectIdGenerators_StringIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      LinkedList<PropertyName> linkedList0 = new LinkedList<PropertyName>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet((JsonParser) null, defaultDeserializationContext_Impl0, linkedList0);
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
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdGenerator<List<PropertyName>> objectIdGenerator0 = (ObjectIdGenerator<List<PropertyName>>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdValueProperty0, (JsonDeserializer<?>) null, jsonDeserializer0);
      ObjectIdValueProperty objectIdValueProperty2 = new ObjectIdValueProperty(objectIdValueProperty1, objectIdReader0.propertyName);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty2.withValueDeserializer((JsonDeserializer<?>) null);
      assertNotSame(settableBeanProperty0, objectIdValueProperty2);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, (PropertyName) null, objectIdGenerators_StringIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertSame(settableBeanProperty0, objectIdValueProperty0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.UUIDGenerator objectIdGenerators_UUIDGenerator0 = new ObjectIdGenerators.UUIDGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(simpleType0, propertyName0, objectIdGenerators_UUIDGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      JsonParser jsonParser0 = jsonFactory0.createParser((char[]) null, (-1126), (-1126));
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeSetAndReturn(jsonParser0, defaultDeserializationContext_Impl0, simpleObjectIdResolver0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}
