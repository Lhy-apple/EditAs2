/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:41 GMT 2023
 */

package com.fasterxml.jackson.databind.module;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimpleAbstractTypeResolver_ESTest extends SimpleAbstractTypeResolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JavaType javaType0 = simpleAbstractTypeResolver0.resolveAbstractType((DeserializationConfig) null, simpleType0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      Class<Object> class1 = Object.class;
      // Undeclared exception!
      try { 
        simpleAbstractTypeResolver0.addMapping(class1, (Class<?>) class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not add mapping from class java.lang.Object since it is not abstract
         //
         verifyException("com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<SimpleObjectIdResolver> class0 = SimpleObjectIdResolver.class;
      // Undeclared exception!
      try { 
        simpleAbstractTypeResolver0.addMapping(class0, (Class<? extends SimpleObjectIdResolver>) class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not add mapping from class to itself
         //
         verifyException("com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      Class<JsonDeserializer> class1 = JsonDeserializer.class;
      // Undeclared exception!
      try { 
        simpleAbstractTypeResolver0.addMapping((Class<JsonDeserializer<LinkedList>>) class0, (Class<? extends JsonDeserializer<LinkedList>>) class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not add mapping from class com.fasterxml.jackson.annotation.ObjectIdResolver to com.fasterxml.jackson.databind.JsonDeserializer, as latter is not a subtype of former
         //
         verifyException("com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<TypeIdResolver> class0 = TypeIdResolver.class;
      Class<MinimalClassNameIdResolver> class1 = MinimalClassNameIdResolver.class;
      SimpleAbstractTypeResolver simpleAbstractTypeResolver1 = simpleAbstractTypeResolver0.addMapping(class0, (Class<? extends TypeIdResolver>) class1);
      assertSame(simpleAbstractTypeResolver0, simpleAbstractTypeResolver1);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      JavaType javaType0 = simpleAbstractTypeResolver0.findTypeMapping((DeserializationConfig) null, simpleType0);
      assertNull(javaType0);
  }
}