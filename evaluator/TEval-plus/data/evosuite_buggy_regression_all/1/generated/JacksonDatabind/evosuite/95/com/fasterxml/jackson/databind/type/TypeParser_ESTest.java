/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:43:32 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeParser;
import com.fasterxml.jackson.databind.util.LRUMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeParser_ESTest extends TypeParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser.MyTokenizer typeParser_MyTokenizer0 = new TypeParser.MyTokenizer("com.fasterxml.jackson.databind.type.TypeParser");
      try { 
        typeParser0.parseTypes(typeParser_MyTokenizer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'com.fasterxml.jackson.databind.type.TypeParser' (remaining: ''): Unexpected end-of-string
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser.MyTokenizer typeParser_MyTokenizer0 = new TypeParser.MyTokenizer("com.fasterxml.jackson.databind.type.TypeParser");
      typeParser_MyTokenizer0._pushbackToken = "com.fasterxml.jackson.databind.type.TypeParser";
      try { 
        typeParser0.parseTypes(typeParser_MyTokenizer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'com.fasterxml.jackson.databind.type.TypeParser' (remaining: ''): Unexpected token 'com.fasterxml.jackson.databind.type.TypeParser', expected ',' or '>')
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser typeParser1 = typeParser0.withFactory((TypeFactory) null);
      assertNotSame(typeParser1, typeParser0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(68, 68);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser typeParser1 = typeParser0.withFactory(typeFactory0);
      assertSame(typeParser1, typeParser0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      JavaType javaType0 = typeParser0.parse("com.fasterxml.jackson.databind.PropertyNamingStrategy$SnakeCaseStrategy");
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      try { 
        typeParser0.parse("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '' (remaining: ''): Unexpected end-of-string
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(68, 68);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser.MyTokenizer typeParser_MyTokenizer0 = new TypeParser.MyTokenizer("[J<t:4mKF4i=bl|xNK@");
      try { 
        typeParser0.parseTypes(typeParser_MyTokenizer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '[J<t:4mKF4i=bl|xNK@' (remaining: ''): Can not locate class 't:4mKF4i=bl|xNK@', problem: Class 't:4mKF4i=bl|xNK@.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeParser.MyTokenizer typeParser_MyTokenizer0 = new TypeParser.MyTokenizer("");
      try { 
        typeParser0.parseTypes(typeParser_MyTokenizer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '' (remaining: ''): Unexpected end-of-string
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TypeParser typeParser0 = new TypeParser((TypeFactory) null);
      // Undeclared exception!
      try { 
        typeParser0.parse("com.fasterxml.jackson.databind.PropertyNamingStrategy$SnakeCaseStrategy");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }
}
