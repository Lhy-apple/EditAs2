/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:52:38 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.KeyDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.ReadableObjectId;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultDeserializationContext_ESTest extends DefaultDeserializationContext_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectReader objectReader0 = objectMapper0.reader(jsonNodeFactory0);
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      ObjectReader objectReader1 = objectReader0.forType((JavaType) simpleType0);
      assertNotSame(objectReader0, objectReader1);
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
      ReadableObjectId readableObjectId0 = defaultDeserializationContext_Impl0.findObjectId((Object) beanDeserializerFactory0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      ReadableObjectId readableObjectId1 = defaultDeserializationContext_Impl0.findObjectId((Object) beanDeserializerFactory0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      assertSame(readableObjectId1, readableObjectId0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ReadableObjectId readableObjectId0 = defaultDeserializationContext_Impl0.findObjectId((Object) objectIdGenerators_IntSequenceGenerator0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      ReadableObjectId readableObjectId1 = defaultDeserializationContext_Impl0.findObjectId((Object) readableObjectId0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      assertNotSame(readableObjectId1, readableObjectId0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      defaultDeserializationContext_Impl0.findObjectId((Object) beanDeserializerFactory0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0);
      defaultDeserializationContext_Impl0.checkUnresolvedObjectId();
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
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
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
      JsonDeserializer<Object> jsonDeserializer0 = defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, (Object) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<ReadableObjectId> class0 = ReadableObjectId.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, (-60));
      JsonDeserializer<Object> jsonDeserializer0 = defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, coreXMLDeserializers_Std0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
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
  public void test11()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.deserializerInstance((Annotated) null, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned Class java.lang.String; expected Class<JsonDeserializer>
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned Class com.fasterxml.jackson.databind.ext.CoreXMLDeserializers$Std; expected Class<KeyDeserializer>
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      KeyDeserializer keyDeserializer0 = defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, (Object) null);
      assertNull(keyDeserializer0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      KeyDeserializer keyDeserializer0 = defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, stdKeyDeserializer0);
      assertNotNull(keyDeserializer0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectReader objectReader0 = objectMapper0.reader(jsonNodeFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.keyDeserializerInstance((Annotated) null, objectReader0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned key deserializer definition of type com.fasterxml.jackson.databind.ObjectReader; expected type KeyDeserializer or Class<KeyDeserializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.deser.DefaultDeserializationContext", e);
      }
  }
}
