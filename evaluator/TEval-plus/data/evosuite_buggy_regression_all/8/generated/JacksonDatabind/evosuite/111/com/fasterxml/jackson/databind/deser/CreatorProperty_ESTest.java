/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:21:13 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.PipedReader;
import java.io.Reader;
import java.lang.annotation.Annotation;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorProperty_ESTest extends CreatorProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 202, javaType0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertEquals(202, settableBeanProperty0.getCreatorIndex());
      assertNotSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withName((PropertyName) null);
      assertEquals(187, settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1315), javaType0, propertyMetadata0);
      try { 
        creatorProperty0.set((Object) null, javaType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, ")h!*.AUBh@z");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-896), annotationIntrospector0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withNullProvider((NullValueProvider) null);
      assertEquals((-896), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      PipedReader pipedReader0 = new PipedReader();
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) pipedReader0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, false, false);
      try { 
        creatorProperty0.deserializeSetAndReturn(filteringParserDelegate0, (DeserializationContext) null, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-758), javaType0, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.getDeclaringClass();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.SettableBeanProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, ")h!*.AUBh@z");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-896), annotationIntrospector0, propertyMetadata0);
      creatorProperty0.markAsIgnorable();
      assertTrue(creatorProperty0.isIgnorable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      Object object1 = creatorProperty0.getInjectableValueId();
      assertNotNull(object1);
      assertEquals(187, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      PropertyName propertyName0 = annotationIntrospector0.findRootName((AnnotatedClass) null);
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-901), annotationMap0, propertyMetadata0);
      String string0 = creatorProperty0.toString();
      assertEquals("[creator property, name ''; inject id '[null]']", string0);
      assertEquals((-901), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-758), javaType0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        creatorProperty0.inject(defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      creatorProperty0.setFallbackSetter((SettableBeanProperty) null);
      assertEquals(187, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, integer0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-758), annotationIntrospector0, propertyMetadata0);
      ObjectIdInfo objectIdInfo0 = ObjectIdInfo.empty();
      ObjectIdReferenceProperty objectIdReferenceProperty0 = new ObjectIdReferenceProperty(creatorProperty0, objectIdInfo0);
      int int0 = objectIdReferenceProperty0.getCreatorIndex();
      assertEquals((-758), int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, ")h!*.AUBh@z");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-896), annotationIntrospector0, propertyMetadata0);
      creatorProperty0.isIgnorable();
      assertEquals((-896), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-758), annotationIntrospector0, propertyMetadata0);
      JsonDeserializer<Object> jsonDeserializer0 = SettableBeanProperty.MISSING_VALUE_DESERIALIZER;
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertEquals((-758), settableBeanProperty0.getCreatorIndex());
      assertSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      creatorProperty0.fixAccess((DeserializationConfig) null);
      assertEquals(187, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-901));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.CreatorProperty", integer0, "com.fasterxml.jackson.databind.deser.CreatorProperty");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-901), (Object) null, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.findInjectableValue((DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<ObjectIdResolver> class1 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      ReferenceType referenceType0 = ReferenceType.construct(class1, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 1104);
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 1104, (Object) null, propertyMetadata0);
      Class<Annotation> class2 = Annotation.class;
      creatorProperty0.getAnnotation(class2);
      assertEquals(1104, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-896));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "com.fasterxml.jackson.databind.deser.BZanDeserializer$1", integer0, "");
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      Class<Annotation> class1 = Annotation.class;
      creatorProperty0.getAnnotation(class1);
      assertEquals(187, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, referenceType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 187, object0, propertyMetadata0);
      ObjectIdReferenceProperty objectIdReferenceProperty0 = new ObjectIdReferenceProperty(creatorProperty0, (ObjectIdInfo) null);
      creatorProperty0._fallbackSetter = (SettableBeanProperty) objectIdReferenceProperty0;
      Stack<PropertyName> stack0 = new Stack<PropertyName>();
      // Undeclared exception!
      try { 
        creatorProperty0.setAndReturn("com.fasterxml.jackson.databind.deser.BZanDeserializer$1", stack0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}