/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:59:44 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.node.POJONode;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = new PropertyName("b Y/3TY8", "b Y/3TY8");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withName((PropertyName) null);
      assertEquals((-1), objectIdValueProperty1.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      POJONode pOJONode0 = new POJONode(propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set((Object) null, pOJONode0);
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
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = mock(ObjectIdGenerator.IdKey.class, new ViolatedAssumptionAnswer());
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn((ObjectIdGenerator.IdKey) null).when(objectIdGenerator0).key(any());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn("h_xcAmxj*+{-iF%").when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = new Object();
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet((JsonParser) null, defaultDeserializationContext_Impl0, object0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ReadableObjectId", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      JsonDeserializer<Annotation> jsonDeserializer0 = (JsonDeserializer<Annotation>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<ObjectIdGenerators.UUIDGenerator> jsonDeserializer1 = (JsonDeserializer<ObjectIdGenerators.UUIDGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer(jsonDeserializer1);
      assertEquals((-1), objectIdValueProperty1.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<Annotation> jsonDeserializer0 = (JsonDeserializer<Annotation>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerators_StringIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = objectIdValueProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = mock(ObjectIdGenerator.IdKey.class, new ViolatedAssumptionAnswer());
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn(objectIdGenerator_IdKey0).when(objectIdGenerator0).key(any());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn("h_xcAmxj*+{-iF%").when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      JsonDeserializer<PipedInputStream> jsonDeserializer1 = (JsonDeserializer<PipedInputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer1, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdReader objectIdReader1 = ObjectIdReader.construct((JavaType) null, (PropertyName) null, objectIdReader0.generator, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) objectIdValueProperty0, objectIdReader0.resolver);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty1.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ObjectIdGenerator<InputStream> objectIdGenerator0 = (ObjectIdGenerator<InputStream>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JsonDeserializer<String> jsonDeserializer1 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader1 = new ObjectIdReader((JavaType) null, (PropertyName) null, objectIdReader0.generator, jsonDeserializer1, objectIdValueProperty0, simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      Integer integer0 = new Integer((-1186));
      // Undeclared exception!
      try { 
        objectIdValueProperty1.setAndReturn(beanDeserializerFactory0, integer0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}