/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:25:36 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.KeyDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.ReadableObjectId;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.Annotated;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultDeserializationContext_ESTest extends DefaultDeserializationContext_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.reader((Class<?>) class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.with(beanDeserializerFactory0);
      assertNotSame(defaultDeserializationContext_Impl0, defaultDeserializationContext0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.copy();
      assertNotSame(defaultDeserializationContext_Impl0, defaultDeserializationContext0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      defaultDeserializationContext_Impl0.findObjectId((Object) beanDeserializerFactory0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      defaultDeserializationContext_Impl0.checkUnresolvedObjectId();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      ReadableObjectId readableObjectId0 = defaultDeserializationContext_Impl0.findObjectId((Object) objectIdGenerators_IntSequenceGenerator0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      ReadableObjectId readableObjectId1 = defaultDeserializationContext_Impl0.findObjectId((Object) objectIdGenerators_IntSequenceGenerator0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      assertSame(readableObjectId1, readableObjectId0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      ReadableObjectId readableObjectId0 = defaultDeserializationContext_Impl0.findObjectId((Object) objectIdGenerators_IntSequenceGenerator0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      ReadableObjectId readableObjectId1 = defaultDeserializationContext_Impl0.findObjectId((Object) readableObjectId0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      assertNotSame(readableObjectId0, readableObjectId1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      defaultDeserializationContext_Impl0.checkUnresolvedObjectId();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, (-955));
      JsonDeserializer<Object> jsonDeserializer0 = defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, coreXMLDeserializers_Std0);
      assertSame(coreXMLDeserializers_Std0, jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, beanDeserializerFactory0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned deserializer definition of type com.fasterxml.jackson.databind.deser.BeanDeserializerFactory; expected type JsonDeserializer or Class<JsonDeserializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.TYPE;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned Class int; expected Class<JsonDeserializer>
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      StdKeyDeserializer stdKeyDeserializer1 = (StdKeyDeserializer)defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, stdKeyDeserializer0);
      assertEquals(14, StdKeyDeserializer.TYPE_URL);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      KeyDeserializer keyDeserializer0 = defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, (Object) null);
      assertNull(keyDeserializer0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned Class java.lang.String; expected Class<KeyDeserializer>
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, defaultDeserializationContext_Impl0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned key deserializer definition of type com.fasterxml.jackson.databind.deser.DefaultDeserializationContext$Impl; expected type KeyDeserializer or Class<KeyDeserializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }
}
