/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:38:13 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.NullNode;
import java.io.InputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringArrayDeserializer_ESTest extends StringArrayDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      JsonParser jsonParser0 = jsonFactory0.createParser((InputStream) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      NullNode nullNode0 = NullNode.getInstance();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(nullNode0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ArrayNode arrayNode1 = arrayNode0.insert(56320, "com.fasterxml.jackson.databind.node.TreeTraversingParser");
      JsonParser jsonParser0 = arrayNode1.traverse();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ArrayNode arrayNode1 = arrayNode0.insert(56320, "com.fasterLml.jackson.databinD.node.TreeTraversingParser");
      JsonParser jsonParser0 = arrayNode1.traverse();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      Integer integer0 = new Integer((-771));
      arrayNode0.add(integer0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      arrayNode0.add((Integer) null);
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      NullNode nullNode0 = NullNode.getInstance();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(nullNode0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonDeserializer<CreatorProperty> jsonDeserializer0 = (JsonDeserializer<CreatorProperty>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null, (Object) null, (Object) null, (Object) null, (Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer(jsonDeserializer0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      stringArrayDeserializer0._deserializeCustom(jsonParser0, deserializationContext0);
      // Undeclared exception!
      stringArrayDeserializer0._deserializeCustom(jsonParser0, deserializationContext0);
  }
}