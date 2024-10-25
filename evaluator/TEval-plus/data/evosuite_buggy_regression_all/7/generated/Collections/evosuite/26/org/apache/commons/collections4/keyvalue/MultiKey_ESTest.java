/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:32:02 GMT 2023
 */

package org.apache.commons.collections4.keyvalue;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Array;
import org.apache.commons.collections4.keyvalue.MultiKey;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultiKey_ESTest extends MultiKey_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MultiKey<String> multiKey0 = null;
      try {
        multiKey0 = new MultiKey<String>((String[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The array of keys must not be null
         //
         verifyException("org.apache.commons.collections4.keyvalue.MultiKey", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      String[] stringArray0 = new String[1];
      MultiKey<String> multiKey0 = new MultiKey<String>(stringArray0);
      String[] stringArray1 = multiKey0.getKeys();
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MultiKey<String> multiKey0 = new MultiKey<String>("", "");
      String string0 = multiKey0.toString();
      assertEquals("MultiKey[, ]", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MultiKey<Object>[] multiKeyArray0 = (MultiKey<Object>[]) Array.newInstance(MultiKey.class, 0);
      MultiKey<MultiKey<Object>> multiKey0 = new MultiKey<MultiKey<Object>>(multiKeyArray0, true);
      multiKey0.hashCode();
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      MultiKey<String> multiKey0 = new MultiKey<String>("org.apache.commons.collections4.keyvalue.MultiKey", "org.apache.commons.collections4.keyvalue.MultiKey", "org.apache.commons.collections4.keyvalue.MultiKey", "org.apache.commons.collections4.keyvalue.MultiKey", "org.apache.commons.collections4.keyvalue.MultiKey");
      boolean boolean0 = multiKey0.equals("org.apache.commons.collections4.keyvalue.MultiKey");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Object[] objectArray0 = new Object[0];
      MultiKey<Object> multiKey0 = new MultiKey<Object>(objectArray0);
      MultiKey<MultiKey<Object>> multiKey1 = new MultiKey<MultiKey<Object>>(multiKey0, multiKey0, multiKey0);
      Object object0 = multiKey1.getKey(2);
      boolean boolean0 = multiKey0.equals(object0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Integer integer0 = new Integer(1027);
      MultiKey<Integer> multiKey0 = new MultiKey<Integer>(integer0, integer0, integer0);
      int int0 = multiKey0.size();
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Integer integer0 = new Integer((-146));
      MultiKey<Integer> multiKey0 = new MultiKey<Integer>(integer0, integer0, integer0);
      MultiKey<MultiKey<Integer>> multiKey1 = new MultiKey<MultiKey<Integer>>(multiKey0, multiKey0, multiKey0, multiKey0);
      boolean boolean0 = multiKey1.equals(multiKey0);
      assertFalse(boolean0);
  }
}
