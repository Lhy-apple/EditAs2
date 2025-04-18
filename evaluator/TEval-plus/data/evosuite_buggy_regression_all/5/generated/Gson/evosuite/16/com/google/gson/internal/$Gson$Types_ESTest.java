/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:56:16 GMT 2023
 */

package com.google.gson.internal;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.WildcardType;
import java.util.Properties;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class $Gson$Types_ESTest extends $Gson$Types_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      Type type0 = .Gson.Types.getSupertype(wildcardType0, class0, class0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(genericArrayType0);
      Class class2 = (Class).Gson.Types.resolve(genericArrayType0, class0, class1);
      assertEquals("class [Ljava.lang.String;", class2.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      Properties properties0 = new Properties();
      Object object0 = properties0.put(parameterizedType0, parameterizedType0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[1];
      typeArray0[0] = (Type) class0;
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(parameterizedType0);
      Type type0 = .Gson.Types.canonicalize(wildcardType0);
      boolean boolean0 = .Gson.Types.equals((Type) wildcardType0, type0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Properties properties0 = new Properties();
      Object object0 = properties0.put(genericArrayType0, class0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(wildcardType0);
      assertTrue(wildcardType1.equals((Object)wildcardType0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) wildcardType1, (Type) wildcardType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.supertypeOf(wildcardType0);
      assertTrue(wildcardType1.equals((Object)wildcardType0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(genericArrayType0);
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class1);
      assertNotNull(wildcardType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      GenericArrayType genericArrayType1 = .Gson.Types.arrayOf(genericArrayType0);
      boolean boolean0 = .Gson.Types.equal(genericArrayType0, genericArrayType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner((Type) null, (Type) null, typeArray0);
      // Undeclared exception!
      try { 
        .Gson.Types.getRawType(parameterizedType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.internal.$Gson$Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(wildcardType0);
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      // Undeclared exception!
      try { 
        .Gson.Types.getRawType(genericArrayType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected a Class, ParameterizedType, or GenericArrayType, but <null> is of type null
         //
         verifyException("com.google.gson.internal.$Gson$Types", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(parameterizedType0, parameterizedType0, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType0, (Type) parameterizedType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Properties properties0 = new Properties();
      boolean boolean0 = .Gson.Types.equal((Object) null, properties0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equal(wildcardType0, wildcardType1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) class0, (Type) wildcardType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner((Type) null, (Type) null, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType0, (Type) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(class0, parameterizedType0, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType0, (Type) parameterizedType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[1];
      typeArray0[0] = (Type) class0;
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      Type[] typeArray1 = new Type[4];
      typeArray1[0] = (Type) parameterizedType0;
      typeArray1[1] = (Type) class0;
      typeArray1[2] = (Type) parameterizedType0;
      typeArray1[3] = (Type) class0;
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray1);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType1, typeArray1[0]);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      GenericArrayType genericArrayType1 = .Gson.Types.arrayOf(class0);
      boolean boolean0 = .Gson.Types.equal(genericArrayType0, genericArrayType1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      boolean boolean0 = .Gson.Types.equals((Type) genericArrayType0, (Type) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      boolean boolean0 = .Gson.Types.equals((Type) null, (Type) genericArrayType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) wildcardType0, (Type) class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type type0 = .Gson.Types.canonicalize(class0);
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(type0);
      WildcardType wildcardType1 = .Gson.Types.supertypeOf(class0);
      boolean boolean0 = .Gson.Types.equal(wildcardType0, wildcardType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      int int0 = .Gson.Types.hashCodeOrZero((Object) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      String string0 = .Gson.Types.typeToString(genericArrayType0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      String string0 = .Gson.Types.typeToString(class0);
      assertEquals("java.util.Properties", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<String> class0 = String.class;
      Class<Properties> class1 = Properties.class;
      Type[] typeArray0 = .Gson.Types.getMapKeyAndValueTypes(class0, class1);
      assertEquals(2, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<String> class1 = String.class;
      Class class2 = (Class).Gson.Types.getGenericSupertype(class0, class1, class0);
      assertFalse(class2.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Class<Properties> class1 = Properties.class;
      Class class2 = (Class).Gson.Types.getGenericSupertype(class0, class0, class1);
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<String> class0 = String.class;
      Type type0 = .Gson.Types.getArrayComponentType(class0);
      assertNull(type0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Object> class0 = Object.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Class class1 = (Class).Gson.Types.getArrayComponentType(genericArrayType0);
      assertFalse(class1.isEnum());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = .Gson.Types.getMapKeyAndValueTypes(class0, class0);
      assertEquals(2, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<String> class0 = String.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Type type0 = .Gson.Types.getSupertype(genericArrayType0, class0, class0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[1];
      typeArray0[0] = (Type) class0;
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      Type type0 = .Gson.Types.resolve(class0, class0, parameterizedType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      Type type0 = .Gson.Types.resolve(wildcardType0, class0, wildcardType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner((Type) null, class0, typeArray0);
      assertNotNull(parameterizedType0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner((Type) null, (Type) null, typeArray0);
      boolean boolean0 = .Gson.Types.equal(parameterizedType0, (Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Class<Object> class0 = Object.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      boolean boolean0 = .Gson.Types.equal(genericArrayType0, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equal(wildcardType0, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      Properties properties0 = new Properties();
      Object object0 = properties0.put(wildcardType0, wildcardType0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      Properties properties0 = new Properties();
      Object object0 = properties0.put(wildcardType0, class0);
      assertNull(object0);
  }
}
