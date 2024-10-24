/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:36:12 GMT 2023
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
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.File;
import java.time.chrono.MinguoEra;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringArrayDeserializer_ESTest extends StringArrayDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "JSON", true, class0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, asPropertyTypeDeserializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("JSON", "JSON");
      JsonParser jsonParser0 = jsonFactory0.createParser(file0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ArrayNode arrayNode1 = arrayNode0.add(0.0F);
      JsonParser jsonParser0 = arrayNode1.traverse();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("JSON", "JSON");
      JsonParser jsonParser0 = jsonFactory0.createParser(file0);
      JsonDeserializer<Class<MinguoEra>> jsonDeserializer0 = (JsonDeserializer<Class<MinguoEra>>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null, (Object) null, (Object) null, (Object) null, (Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer(jsonDeserializer0);
      // Undeclared exception!
      stringArrayDeserializer0._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
  }
}
