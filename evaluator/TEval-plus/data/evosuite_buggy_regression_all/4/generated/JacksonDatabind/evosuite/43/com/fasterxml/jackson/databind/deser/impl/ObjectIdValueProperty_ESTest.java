/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:42:27 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.construct("WXXQE(7E[mO*h-~^!$");
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) null, (JsonDeserializer<?>) null, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withName(propertyName0);
      assertFalse(objectIdValueProperty1.isRequired());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set((Object) null, (Object) null);
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
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<ObjectIdResolver> objectIdGenerator0 = (ObjectIdGenerator<ObjectIdResolver>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class1 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class1);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.construct("WXXQE(7E[mO*h-~^!$");
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<Annotation> jsonDeserializer1 = (JsonDeserializer<Annotation>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer(jsonDeserializer1);
      assertNotSame(objectIdValueProperty1, objectIdValueProperty0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      Integer integer0 = Integer.valueOf((-1));
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(integer0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet((JsonParser) null, (DeserializationContext) null, objectIdReader0);
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
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<Annotation> jsonDeserializer0 = (JsonDeserializer<Annotation>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      objectIdValueProperty0.deserializeAndSet((JsonParser) null, (DeserializationContext) null, propertyMetadata0);
      assertNull(objectIdValueProperty0.getManagedReferenceName());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      PropertyName propertyName0 = PropertyName.construct("WXXQE(7E[mO*h`-~^!$");
      ObjectIdGenerator<Annotation> objectIdGenerator0 = (ObjectIdGenerator<Annotation>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) simpleType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      JsonDeserializer<Integer> jsonDeserializer1 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader1 = ObjectIdReader.construct((JavaType) simpleType0, objectIdReader0.propertyName, (ObjectIdGenerator<?>) objectIdGenerators_StringIdGenerator0, (JsonDeserializer<?>) jsonDeserializer1, (SettableBeanProperty) objectIdValueProperty0, (ObjectIdResolver) null);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      SimpleModule simpleModule0 = new SimpleModule("JSON");
      // Undeclared exception!
      try { 
        objectIdValueProperty1.setAndReturn(objectIdGenerators_StringIdGenerator0, simpleModule0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}
