/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:40:33 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.impl.InnerClassProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InnerClassProperty_ESTest extends InnerClassProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'M\"eP o8pVi)<y", "'M\"eP o8pVi)<y");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("'M\"eP o8pVi)<y");
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, (DeserializationContext) null, creatorProperty0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("'M\"eP o8pVi)<y", "'M\"eP o8pVi)<y");
      JavaType javaType0 = TypeFactory.unknownType();
      Class<LongNode> class0 = LongNode.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "'M\"eP o8pVi)<y", true, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 1556);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 1556, propertyMetadata0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeSetAndReturn((JsonParser) null, (DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.SettableBeanProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty");
      JavaType javaType0 = TypeFactory.unknownType();
      Class<JavaType> class0 = JavaType.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 1556);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, (-1587), propertyName0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      int int0 = innerClassProperty0.getPropertyIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 188);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 188, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.set(annotationMap0, javaType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No fallback setter/field defined: can not use creator property for com.fasterxml.jackson.databind.deser.CreatorProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      innerClassProperty0.assignIndex(100);
      assertEquals(100, innerClassProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("Faile to insantiate class ", "Faile to insantiate class ");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = innerClassProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("Failed to instantiate class ", "Failed to instantiate class ");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.setAndReturn((Object) null, propertyMetadata0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No fallback setter/field defined: can not use creator property for com.fasterxml.jackson.databind.deser.CreatorProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, (-2112));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-2112), annotatedParameter0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertFalse(innerClassProperty1.hasViews());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, (-2112));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-2112), annotatedParameter0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.readResolve();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Missing constructor (broken JDK (de)serialization?)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withName(propertyName0);
      assertFalse(innerClassProperty1.hasViews());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty");
      JavaType javaType0 = TypeFactory.unknownType();
      Class<JavaType> class0 = JavaType.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 1556);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, (-1587), propertyName0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      AnnotatedMember annotatedMember0 = innerClassProperty0.getMember();
      assertSame(annotatedParameter0, annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty");
      JavaType javaType0 = TypeFactory.unknownType();
      Class<JavaType> class0 = JavaType.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 1556);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 1556, javaType0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty");
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, (DeserializationContext) null, annotatedParameter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 248);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 248, annotationMap0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.writeReplace();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null constructor not allowed
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotatedConstructor", e);
      }
  }
}